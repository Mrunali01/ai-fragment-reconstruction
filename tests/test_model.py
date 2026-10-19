"""Verify model computations and gradients without training or measuring accuracy."""

import io
import math

from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from jigsaw_ai.model_loss import pair_loss
from jigsaw_ai.siamese import SiameseCNN
from jigsaw_ai.tensor_data import collate_pairs, image_tensor


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def reproducible_initialization():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        yield


def sample(label=1, direction=0):
    return {"first": Image.new("RGB", (64, 64), (255, 128, 0)),
            "second": Image.new("RGB", (64, 64), (0, 20, 255)),
            "compatible": label, "direction": direction}


def test_tensor_channel_order_range_and_storage():
    image = Image.new("RGB", (64, 64), (255, 128, 0))
    tensor = image_tensor(image)
    assert tensor.shape == (3, 64, 64) and tensor.dtype == torch.float32
    assert tensor.is_contiguous()
    assert torch.allclose(tensor[:, 0, 0], torch.tensor([1, 128 / 255, 0]))
    image.putpixel((0, 0), (0, 0, 0))
    assert tensor[0, 0, 0].item() == 1
    with pytest.raises(ValueError, match="RGB 64x64"):
        image_tensor(Image.new("L", (64, 64)))


def test_custom_collation_handles_none_and_dataloader():
    batch = next(iter(DataLoader([sample(), sample(0, None)], batch_size=2, collate_fn=collate_pairs)))
    assert batch["first"].shape == batch["second"].shape == (2, 3, 64, 64)
    assert batch["compatible"].tolist() == [1, 0]
    assert batch["direction"].tolist() == [0, -1]
    assert batch["direction"].dtype == torch.long
    with pytest.raises(ValueError, match="empty batch"):
        collate_pairs([])
    with pytest.raises(ValueError, match="Positive pairs require"):
        collate_pairs([sample(1, None)])
    with pytest.raises(ValueError, match="Negative pairs must"):
        collate_pairs([sample(0, 0)])


@pytest.mark.parametrize("size", [1, 4])
def test_forward_shapes_and_shared_encoder(size):
    model = SiameseCNN()
    calls = []
    handle = model.encoder.register_forward_hook(lambda module, args, out: calls.append(id(module)))
    outputs = model(torch.rand(size, 3, 64, 64), torch.rand(size, 3, 64, 64))
    handle.remove()
    assert calls == [id(model.encoder), id(model.encoder)]
    assert outputs["compatibility_logits"].shape == (size,)
    assert outputs["direction_logits"].shape == (size, 4)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_fusion_preserves_first_and_second_order():
    model = SiameseCNN()
    a, b = torch.zeros(1, 3, 64, 64), torch.ones(1, 3, 64, 64)
    fused = []
    handle = model.pair_head.register_forward_pre_hook(lambda module, args: fused.append(args[0].detach().clone()))
    model(a, b)
    model(b, a)
    handle.remove()
    assert torch.equal(fused[0][:, :128], fused[1][:, 128:])
    assert torch.equal(fused[0][:, 128:], fused[1][:, :128])
    assert not torch.equal(fused[0], fused[1])


def test_mixed_loss_matches_known_values_and_masks_negative_direction():
    binary = torch.zeros(2, requires_grad=True)
    directional = torch.zeros(2, 4, requires_grad=True)
    outputs = {"compatibility_logits": binary, "direction_logits": directional}
    losses = pair_loss(outputs, torch.tensor([1.0, 0.0]), torch.tensor([2, -1]))
    assert losses["compatibility"].item() == pytest.approx(math.log(2))
    assert losses["direction"].item() == pytest.approx(math.log(4))
    assert losses["loss"].item() == pytest.approx(math.log(2) + math.log(4))
    losses["loss"].backward()
    assert directional.grad[0].abs().sum() > 0
    assert torch.equal(directional.grad[1], torch.zeros(4))
    changed = {"compatibility_logits": binary.detach(),
               "direction_logits": torch.tensor([[0., 0., 0., 0.], [100., -100., 50., -50.]])}
    assert pair_loss(changed, torch.tensor([1.0, 0.0]), torch.tensor([2, -1]))["direction"].item() == losses["direction"].item()


@pytest.mark.parametrize("labels,directions", [([0., 0.], [-1, -1]), ([1., 1.], [0, 3])])
def test_single_class_batches_backward(labels, directions):
    model = SiameseCNN()
    a, b = torch.rand(2, 3, 64, 64, requires_grad=True), torch.rand(2, 3, 64, 64, requires_grad=True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    outputs = model(a, b)
    losses = pair_loss(outputs, torch.tensor(labels), torch.tensor(directions))
    losses["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    if labels[0] == 0:
        assert losses["direction"].item() == 0
        assert torch.equal(model.direction_head.weight.grad, torch.zeros_like(model.direction_head.weight))


def test_extreme_logits_are_numerically_stable():
    outputs = {"compatibility_logits": torch.tensor([1000., -1000.], requires_grad=True),
               "direction_logits": torch.tensor([[1000., -1000., 0., 0.], [0., 0., 0., 0.]], requires_grad=True)}
    losses = pair_loss(outputs, torch.tensor([0., 1.]), torch.tensor([-1, 1]))
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert torch.isfinite(outputs["compatibility_logits"].grad).all()


def test_negative_only_zero_does_not_overflow_for_large_direction_logits():
    outputs = {"compatibility_logits": torch.zeros(2, requires_grad=True),
               "direction_logits": torch.full((2, 4), 1e38, requires_grad=True)}
    losses = pair_loss(outputs, torch.zeros(2), torch.full((2,), -1, dtype=torch.long))
    assert losses["direction"].item() == 0
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert torch.equal(outputs["direction_logits"].grad, torch.zeros(2, 4))


def test_invalid_inputs_and_loss_targets_fail_clearly():
    model = SiameseCNN()
    with pytest.raises(ValueError, match="matching nonempty"):
        model(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))
    with pytest.raises(ValueError, match="floating dtype"):
        model(torch.zeros(1, 3, 64, 64, dtype=torch.uint8), torch.zeros(1, 3, 64, 64, dtype=torch.uint8))
    invalid = torch.full((1, 3, 64, 64), float("nan"))
    with pytest.raises(ValueError, match="finite values"):
        model(invalid, invalid)
    outputs = {"compatibility_logits": torch.zeros(1), "direction_logits": torch.zeros(1, 4)}
    with pytest.raises(ValueError, match="Positive pairs require"):
        pair_loss(outputs, torch.tensor([1.]), torch.tensor([-1]))
    with pytest.raises(ValueError, match="Negative pair directions"):
        pair_loss(outputs, torch.tensor([0.]), torch.tensor([2]))
    with pytest.raises(ValueError, match="0 or 1"):
        pair_loss(outputs, torch.tensor([0.5]), torch.tensor([-1]))
    with pytest.raises(ValueError, match="nonnegative"):
        pair_loss(outputs, torch.tensor([0.]), torch.tensor([-1]), direction_weight=-1)


def test_untrained_state_dict_round_trip_in_memory():
    model = SiameseCNN().eval()
    first, second = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = SiameseCNN().eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    with torch.no_grad():
        original, copied = model(first, second), restored(first, second)
    assert all(torch.equal(original[key], copied[key]) for key in original)

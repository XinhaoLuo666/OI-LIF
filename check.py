"""GPU checks for neurons, model gradients and optional checkpoint archives."""
import argparse
from pathlib import Path
import torch
from oilif import IntegerSpike, OILIF, reset_net, set_offset, scheduled_offset
from runtime import device_from_arg, seed_everything, load_checkpoint, load_model_weights


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    seed_everything(40)
    device = device_from_arg(args.device)
    x = torch.tensor([-.1, 0., .5, 1.5, 4., 4.1], device=device, requires_grad=True)
    IntegerSpike.apply(x, 4).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0., 1., 1., 1., 1., 0.], device=device))
    x = torch.randn(9, 2, 5, device=device) * 5
    node = OILIF(offset=.7).to(device)
    actual, expected = node(x), []
    v = torch.zeros_like(x[0])
    for current in x:
        u = v + (current-v)/2
        q = (u-.7).clamp(0, 4).round()
        expected.append(q)
        v = u-q
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
    single = OILIF(tau=None, step_mode='single', offset=.7).to(device)
    torch.testing.assert_close(single(x), (x-.7).clamp(0, 4).round(), rtol=0, atol=0)
    for schedule in ['log', 'cosine']:
        values = [scheduled_offset(i, 10, 2, schedule) for i in range(21)]
        assert values[0] == 0 and values[10] == values[-1] == 2
        assert all(a <= b for a, b in zip(values, values[1:]))
    from action_recognition.sdtv3 import SDTv3
    from action_recognition.resnet34 import ResNet34
    from audio_classification.model import AudioNet
    from object_detection.model import SpikeYOLO
    from object_detection.loss import v8DetectionLoss
    cases = [(SDTv3(), (2,2,2,32,32)),
             (ResNet34(reset_mode='hard', detach_reset=False), (2,2,2,32,32)),
             (AudioNet(num_classes=20), (2,8,700)),
             (AudioNet(num_classes=35), (2,8,700)),
             (SpikeYOLO(), (2,4,3,64,64))]
    import copy
    three_class_config = copy.deepcopy(cases[-1][0].config)
    three_class_config['nc'] = 3
    cases.append((SpikeYOLO(three_class_config), (2,4,3,64,64)))
    for model, shape in cases:
        model = model.to(device).train()
        set_offset(model, .2)
        x = torch.randn(*shape, device=device)
        output = model(x)
        if isinstance(output, list):
            batch = dict(img=x, batch_idx=torch.tensor([0,1], device=device),
                         cls=torch.tensor([[0.],[float(model.config['nc']-1)]], device=device),
                         bboxes=torch.tensor([[.5,.5,.3,.3],[.4,.4,.2,.2]], device=device))
            loss, _ = v8DetectionLoss(model)(output, batch)
        else:
            loss = torch.nn.functional.cross_entropy(output, torch.tensor([0,1], device=device))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)
        parameters = [p for p in model.parameters() if p.grad is not None and torch.count_nonzero(p.grad)]
        assert parameters, 'All parameter gradients are zero.'
        before = [p.detach().clone() for p in parameters]
        torch.optim.SGD(model.parameters(), lr=.01).step()
        changes = [float((p.detach()-old).abs().max()) for p, old in zip(parameters, before)]
        assert max(changes) > 0, 'Optimizer did not change any parameters.'
        assert all(torch.isfinite(p).all() for p in model.parameters())
        reset_net(model)
        model.eval()
        with torch.inference_mode():
            result = model(x)
            reset_net(model)
            second = model(x)
        if isinstance(result, tuple):
            result, second = result[0], second[0]
        torch.testing.assert_close(result, second, atol=0, rtol=0)
        reset_net(model)
        classes = model.config['nc'] if isinstance(model, SpikeYOLO) else result.shape[-1]
        print(type(model).__name__, f'classes={classes} forward/backward/update/reset PASS',
              f'loss={loss.item():.6f} max_parameter_change={max(changes):.6g}')
    root = Path(__file__).resolve().parent
    for path, model_factory in [('action_recognition/weights/sdtv3_hardvs_oilif.pt', SDTv3),
                        ('object_detection/weights/gen1_oilif.pt', None)]:
        if not (root/path).is_file():
            print(path, 'not supplied; archive reload check skipped')
            continue
        checkpoint = load_checkpoint(root/path)
        model = model_factory() if model_factory is not None else SpikeYOLO(checkpoint['model_config'])
        load_model_weights(model, checkpoint, checkpoint['args']['final_offset'])
        print(path, 'original archive reload PASS')
    print('All checks passed.')


if __name__ == '__main__':
    main()

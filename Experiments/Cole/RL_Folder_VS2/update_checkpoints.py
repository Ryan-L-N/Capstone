import torch

updates = {
    './Current_runs/aggressive/best_model.pt': 19,
    './Current_runs/conservative/best_model.pt': 20,
    './Current_runs/moderate/best_model.pt': 20,
}

for path, new_iter in updates.items():
    try:
        ckpt = torch.load(path, map_location='cpu')
        old_iter = ckpt.get('iteration', 'N/A')
        ckpt['iteration'] = new_iter
        torch.save(ckpt, path)
        name = path.split('/')[-2]
        print(f'{name}: {old_iter} -> {new_iter}')
    except Exception as e:
        print(f'ERROR {path}: {e}')

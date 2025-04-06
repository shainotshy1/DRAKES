from tree_spex import lgboost_fit, lgboost_to_fourier, lgboost_tree_to_fourier, ExactSolver

num_masks = 1000
num_features = 50
max_solution_order = 5

all_masks = np.random.choice(2, size=(num_masks, num_features))
outputs = np.zeros(num_masks)

for j, mask in enumerate(all_masks):
    outputs[j] = mask_model(mask)

print('Fitting XGBoost Models')
best_model, cv_r2 = lgboost_fit(all_masks, outputs)
print(f'CV r2: {cv_r2}')

fourier_dict = lgboost_to_fourier(best_model)
print(f'Num Fourier Coefficients: {len(fourier_dict)}')

fourier_dict_trunc = dict(sorted(fourier_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:2000])
# Solve for the best mask up to max_solution_order
best_mask = ExactSolver(fourier_dict_trunc, maximize=False, max_solution_order=max_solution_order).solve()

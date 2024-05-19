import jax 
from jax import random
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import ShuffleSplit
import flax.linen as nn
import optax

@jax.jit
def mse_ridge(w, x_batched, y_batched, alpha):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
        pred = jnp.inner(w, x)
        return (pred - y)**2 + alpha * jnp.abs(w).sum()
    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0)

@jax.jit
def rmse(w, x_batched, y_batched, alpha):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
        pred = jnp.inner(w, x)
        return (pred - y)**2
    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.sqrt(jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0))

mse_ridge_cv = jax.vmap(mse_ridge, in_axes=(0, 0, 0, None))
mse_ridge_cv_alphas = jax.vmap(mse_ridge_cv, in_axes=(0, None, None, 0))

rmse_cv = jax.vmap(mse_ridge, in_axes=(0, 0, 0, None))
rmse_cv_alphas = jax.vmap(rmse_cv, in_axes=(0, None, None, 0))

def fit_best_alpha_jax(hp, ML_method, Xfeature_train, Xfeature_test, 
                   y_train, y_test, y_test_clean, best_cv_score,
                   train_score, test_score):
    cv = ShuffleSplit(n_splits=hp.num_cross_val, test_size=0.3, random_state=0)

    # model with shared parameters
    alphas = jnp.array([2**(-8), 2**(-7), 2**(-6), 2**(-5)])
    n_features = Xfeature_train.shape[1]
    key = random.key(0)
    w = random.normal(key, (len(alphas), hp.num_cross_val, n_features,)) # Dummy input data
     # Initialization call
    splits = cv.split(Xfeature_train, y_train)
    X_train_cv = np.stack([Xfeature_train[train_ind] for train_ind, _ in splits])
    splits = cv.split(Xfeature_train, y_train)
    Y_train_cv = np.stack([y_train[train_ind] for train_ind, _ in splits])  
    splits = cv.split(Xfeature_train, y_train) 
    X_test_cv = np.stack([Xfeature_train[test_ind] for _, test_ind in splits])
    splits = cv.split(Xfeature_train, y_train)
    Y_test_cv = np.stack([y_train[test_ind] for _, test_ind in splits]) 

    w = lr_fit(w, X_train_cv, Y_train_cv, alphas)
    rmses = rmse_cv_alphas(w, X_test_cv, Y_test_cv, alphas)
    cv_scores = jnp.mean(rmses, axis=1)
    alpha_ind = jnp.argmax(cv_scores)
    best_cv_score = cv_scores[alpha_ind]

    return best_cv_score, alphas[alpha_ind]

@jax.jit
def opt_fun(w, X_cv, y_cv, alphas):
    return jnp.mean(mse_ridge_cv_alphas(w, X_cv, y_cv, alphas))

def lr_fit(w, X_cv, y_cv, alphas):
    tx = optax.adamw(learning_rate=0.01)
    opt_state = tx.init(w)
    loss_grad_fn = jax.value_and_grad(opt_fun)
    for i in range(10001):
        loss_val, grads = loss_grad_fn(w, X_cv, y_cv, alphas)
        updates, opt_state = tx.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
        if i % 100 == 0:
            print('Loss step {}: '.format(i), loss_val)
    return w

def final_fit(X_train, y_train, X_test, y_test, alpha):
    n_features = X_train.shape[1]
    key = 1234
    w_final = random.normal(key, (n_features,))

    tx = optax.adamw(learning_rate=0.01)
    opt_state = tx.init(w_final)
    loss_grad_fn = jax.value_and_grad(mse_ridge)
    for i in range(10001):
        loss_val, grads = loss_grad_fn(w_final, X_train, y_train, alpha)
        updates, opt_state = tx.update(grads, opt_state, w_final)
        w_final = optax.apply_updates(w_final, updates)
        if i % 100 == 0:
            print('Loss step {}: '.format(i), loss_val)

    train_error = rmse(w_final, X_train, y_train, alpha)
    test_error = rmse(w_final, X_test, y_test, alpha)  
    return train_error, test_error
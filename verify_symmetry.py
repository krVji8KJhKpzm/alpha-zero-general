import numpy as np
from othello.OthelloGame import OthelloGame

def check_action_perms_vs_coords(game: OthelloGame):
    n = game.n
    ok = True
    for k, T in enumerate(game._d4_Ts):
        p = np.asarray(game.d4_action_perms[k], dtype=np.int64)
        # 1) pass 固定
        assert p[n*n] == n*n, f"pass must be fixed in perm[{k}]"
        for a in range(n*n):
            x, y = a % n, a // n         # 列优先: a = x*n + y
            x2, y2 = T(x, y, n)
            a2 = y2 * n + x2
            if p[a] != a2:
                print(f"[perm!=coord] k={k}, a={a} -> p[a]={p[a]}, coord={a2}")
                ok = False
                break
    print("✓ action_perms vs coords:", "OK" if ok else "MISMATCH")
    return ok

def check_inverse_perms(game: OthelloGame):
    ok = True
    n = game.n
    idx = np.arange(n*n+1, dtype=np.int64)
    for k in range(4):
        p   = np.asarray(game.d4_action_perms[k], dtype=np.int64)
        inv = np.asarray(game._inv_act_perms[k], dtype=np.int64)
        if not (np.all(idx == p[inv]) and np.all(idx == inv[p])):
            print(f"[inverse] failed at k={k}")
            ok = False
    print("✓ inverse perms:", "OK" if ok else "MISMATCH")
    return ok

def apply_T_to_board_copy(game: OthelloGame, b: np.ndarray, k: int):
    # 用你实现的坐标版拷贝函数（等价于 game._apply_T_to_board）
    return game._apply_T_to_board(b, game._d4_Ts[k])

def check_valids_equivariance(game: OthelloGame, boards):
    ok = True
    n = game.n
    for b in boards:
        val0 = np.asarray(game.getValidMoves(b, 1), dtype=np.int8)  # 原坐标
        for k in range(4):
            bt = apply_T_to_board_copy(game, b, k)
            val_k = np.asarray(game.getValidMoves(bt, 1), dtype=np.int8)
            p = np.asarray(game.d4_action_perms[k], dtype=np.int64)  # 原->变换后
            inv = np.asarray(game._inv_act_perms[k], dtype=np.int64)  # 变换后->原
            if not np.array_equal(val_k[inv], val0):
                print(f"[valids equiv] mismatch at k={k}")
                ok = False
                print("board:\n", b)
                print(f" val0: {val0[:-1].reshape(n,n)}")
                print(f"val_k: {val_k[:-1].reshape(n,n)}")
                exit()
                break
    print("✓ valids equivariance:", "OK" if ok else "MISMATCH")
    return ok

def check_canonical_key(game: OthelloGame, boards):
    ok = True
    for b in boards:
        key, act_perm, inv_perm, kstar = game.canonical_key_and_perms(b)
        bt = apply_T_to_board_copy(game, b, kstar)
        key2 = bt.flatten(order='C').tobytes()  # 列优先
        if key != key2:
            print(f"[canon key] key != key(T_{kstar}(b))")
            ok = False
            break
        # 再检查 act_perm/ inv_perm 与 kstar 的置换一致
        p = np.asarray(game.d4_action_perms[kstar], dtype=np.int64)
        inv = np.asarray(game._inv_act_perms[kstar], dtype=np.int64)
        if not (np.array_equal(np.asarray(act_perm), p) and np.array_equal(np.asarray(inv_perm), inv)):
            print(f"[canon key] perms mismatch at k*={kstar}")
            ok = False
            break
    print("✓ canonical key:", "OK" if ok else "MISMATCH")
    return ok

def check_stabilizer(game: OthelloGame, init_board):
    G0 = game.stabilizer_indices(init_board)
    print("✓ stabilizer size at init:", len(G0), "(expect 4)")
    return len(G0) == 4

def sample_random_positions(game: OthelloGame, steps=20, seed=0):
    rng = np.random.default_rng(seed)
    boards = []
    b = game.getInitBoard()
    player = 1
    for t in range(steps):
        boards.append(b.copy())
        valids = np.asarray(game.getValidMoves(b, player), dtype=np.int8)
        A = game.getActionSize()
        moves = np.where(valids == 1)[0]
        if moves.size == 0:
            # pass
            a = A - 1
        else:
            a = rng.choice(moves)
        nb, np_player = game.getNextState(b, player, int(a))
        b = game.getCanonicalForm(nb, np_player)  # 规范表述
        player = 1
    return boards

if __name__ == "__main__":
    game = OthelloGame(n=8)

    # 1) 动作置换与坐标公式一致
    check_action_perms_vs_coords(game)

    # 2) 逆置换
    check_inverse_perms(game)

    # 3) 初始稳定子
    init_b = game.getInitBoard()
    check_stabilizer(game, init_b)

    # 采样若干随机局面用于后续检查
    boards = [init_b] + sample_random_positions(game, steps=100, seed=42)

    # 4) 合法着法等变性
    check_valids_equivariance(game, boards)

    # 5) 规范键一致性
    check_canonical_key(game, boards)

    print("=== symmetry verification done ===")

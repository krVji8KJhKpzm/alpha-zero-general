import torch
def build_action_perms(n: int = 8):
    A = torch.arange(n * n, dtype=torch.long).reshape(n, n)
    perms = []
    for i in range(1, 5):
        for j in (True, False):
            M = torch.rot90(A, i, dims=(0, 1))
            if j:
                M = torch.flip(M, dims=(1,))
            perms.append(M.reshape(-1))
    P = torch.stack(perms, dim=0)  # (8, n*n)

    invP = torch.empty_like(P)
    base = torch.arange(n * n, dtype=torch.long)
    for t in range(8):
        inv = torch.empty(n * n, dtype=torch.long)
        inv[P[t]] = base
        invP[t] = inv
    return P, invP

def sym_op_single(boards: torch.Tensor, t: int) -> torch.Tensor:
    """按与 get_symmetries 同序的第 t 个对称作用在 (B,8,8) 上"""
    B, H, W = boards.shape
    i = (t // 2) + 1         # 0,1->1; 2,3->2; 4,5->3; 6,7->4
    j_flip = (t % 2 == 0)    # 偶数位带 FLIPLR
    x = torch.rot90(boards, i, dims=(1, 2))
    if j_flip:
        x = torch.flip(x, dims=(2,))
    return x

def test_symmetry_identity_and_shape():
    B, n = 3, 8
    boards = torch.randn(B, n, n)
    def get_symmetries(boards):
        syms = []
        for i in range(1, 5):
            for j in (True, False):
                newB = torch.rot90(boards, i, dims=(1, 2))
                if j:
                    newB = torch.flip(newB, dims=(2,))
                syms.append(newB)
        return torch.stack(syms, dim=1)  # (B,8,8,8)

    out = get_symmetries(boards)
    assert out.shape == (B, 8, n, n)
    # 索引7是恒等（i=4, j=False）
    assert torch.allclose(out[:, 7], boards), "t=7 应为恒等"

# ------------------ 2) 置换表互为逆 ------------------
def test_permutation_inverse():
    perm_fwd, perm_back = build_action_perms(n=8)
    base = torch.arange(64)
    for t in range(8):
        # back[fwd] = id
        assert torch.equal(perm_back[t][perm_fwd[t]], base)
        # fwd[back] = id
        assert torch.equal(perm_fwd[t][perm_back[t]], base)

# ------------------ 3) 与索引网格的“一致性” ------------------
def test_perm_matches_geometry():
    n = 8
    perm_fwd, perm_back = build_action_perms(n=n)
    A = torch.arange(n * n).reshape(n, n)
    for t in range(8):
        M = sym_op_single(A.unsqueeze(0), t).squeeze(0)  # 变换后的索引网格
        M_flat = M.reshape(-1)                            # M_flat[a_t] = a0
        # 几何关系：在新棋盘位置 a_t 处的值应等于原索引 a0
        # 即 M_flat[ perm_fwd[t] ] == base
        base = torch.arange(n * n)
        assert torch.equal(M_flat[perm_fwd[t]], base), f"M_flat: {M_flat[perm_fwd[t]]}, base:{base}"
        # 同理：perm_back 正好等于 M_flat（新→原）
        assert torch.equal(perm_back[t], M_flat)

# ------------------ 4) 动作往返一致（含 pass=64） ------------------
def test_action_roundtrip_with_pass():
    n = 8
    perm_fwd, perm_back = build_action_perms(n=n)
    pass_id = n * n  # 64
    # 随机 1000 个动作，含 pass
    a0 = torch.randint(0, pass_id + 1, (1000,))
    for t in range(8):
        # 原→新
        a_t = a0.clone()
        mask = (a0 != pass_id)
        a_t[mask] = perm_fwd[t, a0[mask]]
        # 新→原
        back = a_t.clone()
        mask2 = (a_t != pass_id)
        back[mask2] = perm_back[t, a_t[mask2]]
        # 往返等于原动作
        assert torch.equal(back, a0)

# ------------------ 5) 与 get_symmetries 的动作对应 ------------------
def test_board_action_alignment():
    B, n = 32, 8
    perm_fwd, _ = build_action_perms(n=n)
    boards = torch.zeros(B, n, n)
    a0 = torch.randint(0, n*n, (B,))
    r0, c0 = a0 // n, a0 % n
    boards[torch.arange(B), r0, c0] = 1.0

    for t in range(8):
        b_t = sym_op_single(boards, t)                 # (B,8,8)
        a_t = perm_fwd[t, a0]                           # (B,)
        r_t, c_t = a_t // n, a_t % n
        picked = b_t[torch.arange(B), r_t, c_t]         # 应该都是 1
        assert torch.allclose(picked, torch.ones(B))

# 运行全部测试（无 pytest 也能跑）
if __name__ == "__main__":
    test_symmetry_identity_and_shape()
    test_permutation_inverse()
    test_perm_matches_geometry()
    test_action_roundtrip_with_pass()
    test_board_action_alignment()
    print("✅ all tests passed.")
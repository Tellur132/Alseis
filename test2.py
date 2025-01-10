import cirq
import numpy as np
from scipy.linalg import eigh


def prepare_largest_eigvec_state(rho: np.ndarray, data_qubits: list[cirq.Qid]) -> cirq.Circuit:
    """
    密度行列 rho の最大固有ベクトルを計算し，それを初期状態として用意する回路を返す関数。
    """
    # 1. rho の固有値分解
    classical_eigvals, classical_eigvecs = eigh(rho)  # 昇順で返ることが多い
    # 2. 最大固有値に対応するベクトルを取り出す
    idx_max = np.argmax(classical_eigvals)  # 最大の固有値のインデックス
    largest_eigvec = classical_eigvecs[:, idx_max]
    # 3. 正規化（念のため）
    largest_eigvec = largest_eigvec / np.linalg.norm(largest_eigvec)
    # 4. Cirq の状態準備チャネルで回路を作る
    #    （Cirq v0.15以降などでサポートされている。）
    prep_circuit = cirq.Circuit()
    prep_circuit.append(cirq.StatePreparationChannel(
        largest_eigvec)(*data_qubits))

    return prep_circuit


def example_run_qpe_with_largest_eigvec(rho: np.ndarray,
                                        simulator: cirq.Simulator,
                                        t: float,
                                        num_ancilla: int,
                                        repetitions: int = 1000):
    """
    'rho' の最大固有ベクトルを初期状態として用意し，QPE を行うデモ関数。
    """

    # === 1) data_qubits, ancilla_qubits の準備
    n_qubits = int(np.log2(rho.shape[0]))
    # QPEの補助ビット (ancilla)
    ancilla_qubits = [cirq.LineQubit(i) for i in range(num_ancilla)]
    # データ用 qubits は後ろに配置
    data_qubits = [cirq.LineQubit(num_ancilla + i) for i in range(n_qubits)]

    # === 2) 「最大固有ベクトルに初期化」する回路
    init_circuit = prepare_largest_eigvec_state(rho, data_qubits)

    # === 3) QPE回路を作る
    # 既存コードの create_qpe_circuit_for_rho() と同じでOK
    qpe_circuit, _, _ = create_qpe_circuit_for_rho(rho, t, num_ancilla)

    # === 4) 回路を合成： (初期化 → QPE)
    # Cirq では単純に加算(+)すると連結した回路になる
    full_circuit = init_circuit + qpe_circuit

    print("=== Combined circuit (Largest eigvec init + QPE) ===")
    print(full_circuit)

    # === 5) シミュレーション
    result = simulator.run(full_circuit, repetitions=repetitions)

    # === 6) 測定結果の集計
    # ここは従来のQPEコードと同様。例として簡略化した書き方にします。
    measured_values = []
    for r in range(repetitions):
        bits = []
        for i in range(num_ancilla):
            # m{i} というキーで測定結果を取得
            bits.append(int(result.measurements[f"m{i}"][r]))
        # bit配列→整数
        phase_int = 0
        for bit in bits:
            phase_int = (phase_int << 1) | bit
        measured_values.append(phase_int)

    # 度数分布
    counts = {}
    for val in measured_values:
        counts[val] = counts.get(val, 0) + 1

    # 最頻値を取る
    best_val = max(counts, key=counts.get)
    estimated_phase = best_val / (2**num_ancilla)
    # QPEで求めるのは lambda = 2π * phase / t
    lambda_est = 2 * np.pi * estimated_phase / t

    print("\n=== QPE Measurement Distribution ===")
    for val, cnt in sorted(counts.items()):
        print(f"Measured {val} (phase={
              val/(2**num_ancilla):.4f}): {cnt} times")
    print(f"Most frequent measure = {best_val}, phase={estimated_phase:.4f}")
    print(f'Estimated eigenvalue λ_est = {lambda_est:.4f} (assuming t={t})')

    return lambda_est


# --------------------------------------------------
# 既存の create_qpe_circuit_for_rho(rho, t, num_ancilla) の例
# （質問文に載っていたコードと同じものを再掲）
def create_exp_i_rho_t_gate(rho, t):
    from scipy.linalg import eigh
    eigenvals, eigenvecs = eigh(rho)
    exp_diag = np.exp(1j * eigenvals * t)
    V = eigenvecs
    Vdag = np.conjugate(V).T
    diag_exp = np.diag(exp_diag)
    op_matrix = V @ diag_exp @ Vdag
    gate = cirq.MatrixGate(op_matrix, name="exp(i*rho*t)")
    return gate, op_matrix


def create_qpe_circuit_for_rho(rho, t, num_ancilla):
    n_qubits = int(np.log2(rho.shape[0]))
    ancilla_qubits = [cirq.LineQubit(i) for i in range(num_ancilla)]
    data_qubits = [cirq.LineQubit(num_ancilla + i) for i in range(n_qubits)]
    circuit = cirq.Circuit()

    # 1) ancillaを |+> に初期化
    circuit.append([cirq.H(q) for q in ancilla_qubits])

    # 2) e^{i rho t} を制御ユニタリとして適用
    from scipy.linalg import eigh
    for i in range(num_ancilla):
        exponentiated_gate, _ = create_exp_i_rho_t_gate(rho, t * (2**i))
        controlled_unitary = exponentiated_gate.controlled()
        circuit.append(controlled_unitary.on(ancilla_qubits[i], *data_qubits))

    # 3) 逆QFT を ancilla_qubits 上に適用
    circuit.append(cirq.inverse(cirq.qft(*ancilla_qubits)))

    # 4) 測定
    circuit.append([cirq.measure(q, key=f'm{i}')
                   for i, q in enumerate(ancilla_qubits)])
    return circuit, ancilla_qubits, data_qubits


# --------------------------------------------------
# もし main() のような関数を作るなら、以下のように利用してください
if __name__ == "__main__":

    # （1）rho を何らかの方法で作る（ここでは適当にサイズ 4×4 の例）
    #      → 本来は「データエンコード」して平均を取ったものを用意
    dummy_rho = np.array([
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.1, 0.0],
        [0.0, 0.1, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.1]
    ], dtype=complex)

    # Cirq のシミュレーター
    simulator = cirq.Simulator()

    # （2）QPE を最大固有ベクトル上でやってみる
    t = 1.0
    num_ancilla = 20
    repetitions = 2000

    lambda_est = example_run_qpe_with_largest_eigvec(
        dummy_rho,
        simulator,
        t=t,
        num_ancilla=num_ancilla,
        repetitions=repetitions
    )
    print("\n=== Final QPE-based estimation on largest-eigvec state ===")
    print(f"lambda_est = {lambda_est:.4f}")

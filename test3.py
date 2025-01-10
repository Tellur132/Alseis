import random
import cirq
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy.linalg import eigh


# =============================================================================
# グローバルに使うパスや設定
# =============================================================================
CONFIG_PATH = 'config.yaml'


# =============================================================================
# 1. 設定ファイルロード & シード設定
# =============================================================================

def load_config(config_path):
    """
    YAML形式の設定ファイルを読み込む関数。
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def set_random_seed(config):
    """
    設定に基づいてシードを固定またはランダムに設定します。
    """
    seed_config = config.get('random_seed', {})
    mode = seed_config.get('mode', 'random')

    if mode == 'fixed':
        seed = seed_config.get('value', 42)
        np.random.seed(seed)
        random.seed(seed)
        # CirqのSimulatorにシードを渡す（cirq.Simulator(seed=...)）
        return seed
    elif mode == 'random':
        return None
    else:
        raise ValueError("random_seed.mode は 'fixed' または 'random' を指定してください。")


# =============================================================================
# 2. データセット読込 & スケーリング
# =============================================================================

def load_and_scale_dataset(dataset_name, method='standard', random_state=None):
    """
    指定されたデータセットを読み込み、指定方法でスケーリングした上で返す関数。
    dataset_name: 'iris', 'wine', 'breast_cancer', 'digits', 'diabetes', 
                  'moons', 'circles', 'blobs', 'high_dim' など
    method: 'standard', 'minmax', 'robust'
    random_state: データ生成時のランダムシード
    """
    if dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    elif dataset_name == 'digits':
        data = load_digits()
    elif dataset_name == 'diabetes':
        data = load_diabetes()
    elif dataset_name == 'moons':
        data = make_moons(n_samples=500, noise=0.05, random_state=random_state)
    elif dataset_name == 'circles':
        data = make_circles(n_samples=500, noise=0.05,
                            factor=0.5, random_state=random_state)
    elif dataset_name == 'blobs':
        data = make_blobs(n_samples=500, centers=3,
                          cluster_std=1.0, random_state=random_state)
    elif dataset_name == 'high_dim':
        data = make_classification(n_samples=500, n_features=10,
                                   n_informative=8, n_redundant=2,
                                   n_classes=2, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # data が Bunch であれば data.data でアクセスできる
    # タプルであれば (features, target) のように扱う
    if hasattr(data, 'data'):
        features = data.data
        labels = data.target if hasattr(data, 'target') else None
    else:
        features = data[0]
        labels = data[1] if len(data) > 1 else None

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method: {method}")

    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels


# =============================================================================
# 3. データを量子回路へエンコードする関数
# =============================================================================

def create_quantum_encoding_circuit(qubits, data, method='amplitude'):
    """
    データを量子状態にエンコードするための回路を作成。
    method: 'amplitude', 'phase', 'angle', 'qrac' など
    """
    circuit = cirq.Circuit()

    if method == 'phase':
        # 位相エンコードの例
        for i, value in enumerate(data):
            val_norm = (value - np.min(data)) / \
                (np.max(data) - np.min(data) + 1e-10) * 2 - 1.0
            angle = 2 * np.pi * val_norm
            if i < len(qubits):
                circuit.append(cirq.rz(angle)(qubits[i]))
    elif method == 'angle':
        # 角度エンコードの例
        for i, value in enumerate(data):
            val_norm = (value - np.min(data)) / \
                (np.max(data) - np.min(data) + 1e-10) * np.pi
            if i < len(qubits):
                circuit.append(cirq.rx(val_norm)(qubits[i]))
    elif method == 'qrac':
        # 2ビットを1量子ビットに詰め込む例
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                x1 = data[i]
                x2 = data[i+1]
                x1 = (x1 - np.min(data)) / \
                    (np.max(data) - np.min(data) + 1e-10) * 2 - 1.0
                x2 = (x2 - np.min(data)) / \
                    (np.max(data) - np.min(data) + 1e-10) * 2 - 1.0
                angle_y = np.arccos(x1)
                angle_z = np.arccos(x2)
                if i // 2 < len(qubits):
                    circuit.append(cirq.ry(2 * angle_y)(qubits[i // 2]))
                    circuit.append(cirq.rz(2 * angle_z)(qubits[i // 2]))
            else:
                # 余る場合
                val = data[i]
                val = (val - np.min(data)) / \
                    (np.max(data) - np.min(data) + 1e-10) * 2 - 1.0
                angle = 2 * np.arccos(val)
                if i // 2 < len(qubits):
                    circuit.append(cirq.ry(angle)(qubits[i // 2]))
    elif method == 'amplitude':
        # 簡易的な amplitude encoding (正規化してRyに対応させる程度)
        # 本来はベクトル全体を正規化してエンコードするが，ここではサンプルとして
        for i, value in enumerate(data):
            val_norm = (value - np.min(data)) / \
                (np.max(data) - np.min(data) + 1e-10) * 2 - 1.0
            angle = 2 * np.arccos(val_norm)
            if i < len(qubits):
                circuit.append(cirq.ry(angle)(qubits[i]))
    else:
        raise ValueError(
            "Invalid encoding method. Choose 'amplitude', 'phase', 'angle', or 'qrac'.")
    return circuit


# =============================================================================
# 4. （前半コード）ブロックエンコードによる e^{i rho t} を作りQPEする関数群
# =============================================================================

def create_exp_i_rho_t_gate(rho, t):
    """
    密度行列 rho (サイズ 2^n x 2^n) の対角化を使って
      e^{i rho t} = U * diag(e^{i lambda_k t}) * U^dagger
    を行列表現で作り、それを cirq のゲート (MatrixGate) に変換する。
    """
    # rho の固有値分解
    eigenvals, eigenvecs = eigh(rho)
    exp_diag = np.exp(1j * eigenvals * t)
    V = eigenvecs
    Vdag = np.conjugate(V).T
    diag_exp = np.diag(exp_diag)
    op_matrix = V @ diag_exp @ Vdag  # (2^n x 2^n)

    num_qubits = int(np.log2(rho.shape[0]))
    gate = cirq.MatrixGate(op_matrix, name="exp(i*rho*t)")
    return gate, op_matrix


def create_qpe_circuit_for_rho(rho, t, num_ancilla):
    """
    ブロックエンコードされた e^{i rho t} を制御ユニタリとして使った QPE 回路を作成
    ancilla_qubits: フェーズ推定用の補助量子ビット
    data_qubits: 実際に e^{i rho t} が作用する量子ビット
    """
    n_qubits = int(np.log2(rho.shape[0]))
    # QPE用の補助量子ビット
    ancilla_qubits = [cirq.LineQubit(i) for i in range(num_ancilla)]
    data_qubits = [cirq.LineQubit(num_ancilla + i) for i in range(n_qubits)]

    circuit = cirq.Circuit()

    # 1. ancilla を |+> に初期化
    circuit.append([cirq.H(q) for q in ancilla_qubits])

    # 2. e^{i rho t} を 2^i 倍して制御ユニタリ適用
    for i in range(num_ancilla):
        exponentiated_gate, _ = create_exp_i_rho_t_gate(rho, t * (2**i))
        controlled_unitary = exponentiated_gate.controlled()
        circuit.append(controlled_unitary.on(ancilla_qubits[i], *data_qubits))

    # 3. 逆QFT (補助ビット上)
    circuit.append(cirq.inverse(cirq.qft(*ancilla_qubits)))

    # 4. 測定
    circuit.append([cirq.measure(q, key=f'm{i}')
                   for i, q in enumerate(ancilla_qubits)])

    return circuit, ancilla_qubits, data_qubits


def perform_qpca_qpe_blockencoding(scaled_data, simulator, config):
    """
    (前半コードの方式)
    1. 多数サンプルの平均から密度行列 rho を作る
    2. QPE (ブロックエンコード版) で rho の固有値を推定
    3. クラシカルな固有値分解とも比較
    """
    n_features = scaled_data.shape[1]
    n_qubits = int(np.log2(n_features))
    if 2**n_qubits != n_features:
        raise ValueError("特徴量数が 2^n である必要があります。")

    # サンプルごとに量子状態作成 → |psi_j><psi_j| の和
    rho = np.zeros((n_features, n_features), dtype=complex)
    qubits_for_data = [cirq.LineQubit(i) for i in range(n_qubits)]

    # 1サンプルずつ回路シミュレーションし，最終状態から密度行列を作る
    for sample in scaled_data:
        circuit = create_quantum_encoding_circuit(
            qubits_for_data, sample, method=config.get('encoding_method', 'amplitude'))
        result = simulator.simulate(circuit)
        state = result.final_state_vector
        dm = np.outer(state, np.conjugate(state))
        rho += dm

    rho /= len(scaled_data)

    # QPE実行
    t = config.get("time_parameter", 1.0)
    num_ancilla = config.get("num_ancilla", 3)
    qpe_circuit, ancilla_qubits, data_qubits = create_qpe_circuit_for_rho(
        rho, t, num_ancilla)

    if config.get('print_circuit', False):
        print("=== QPE Circuit (block-encoding) ===")
        print(qpe_circuit)

    repetitions = config.get("repetitions", 100)
    result = simulator.run(qpe_circuit, repetitions=repetitions)

    # 測定結果をビット列として集計
    measured_values = []
    for r in range(repetitions):
        bits = [int(result.measurements[f"m{i}"][r])
                for i in range(num_ancilla)]
        phase_int = 0
        for bit in bits:
            phase_int = (phase_int << 1) | bit
        measured_values.append(phase_int)

    # 度数分布
    counts = {}
    for val in measured_values:
        counts[val] = counts.get(val, 0) + 1

    best_val = max(counts, key=counts.get)
    estimated_phase = best_val / (2**num_ancilla)
    lambda_est = 2 * np.pi * estimated_phase / t

    if config.get('print_eigenvalues', False):
        print("=== QPE Measurement Distribution (block-encoding) ===")
        for val, cnt in sorted(counts.items()):
            print(f"Measured {val} (phase={
                  val/(2**num_ancilla):.4f}): {cnt} times")
        print(f"Most frequent measure = {
              best_val}, phase={estimated_phase:.4f}")
        print(f"Estimated eigenvalue λ_est = {
              lambda_est:.4f} (assuming t={t})")

    # 古典的にrhoを固有値分解
    classical_eigvals, _ = eigh(rho)
    classical_eigvals_sorted = np.sort(classical_eigvals)[::-1]

    return classical_eigvals_sorted, lambda_est, rho


def extract_all_eigenvalues_from_qpe_measurements(
    counts,
    num_ancilla,
    t,
    repetitions,
    phase_clustering_tol=None
):
    """
    QPEの測定ヒストグラム（counts）から，
      bit列 -> 推定固有値 λ_est
    を計算し，その出現頻度を固有値の「重み」として抽出する。

    params:
      counts: { bit_int_value : count } の辞書
      num_ancilla: アンシラ量子ビット数
      t: e^{i rho t} における時間パラメータ
      repetitions: 測定回数（sum(counts.values()) のはず）
      phase_clustering_tol: 近い位相をまとめる際の許容誤差（ラジアン）。Noneならクラスタリングしない。

    return:
      eigenvalues_list: [(lambda_est, probability), ... ] のリスト
        probability は出現頻度 / repetitions
    """
    # bit列 -> 推定固有値
    raw_estimates = []
    for val, cnt in counts.items():
        phase_float = val / (2**num_ancilla)
        lambda_est = (2 * np.pi * phase_float) / t  # λ = 2π*phase / t
        weight = cnt / repetitions  # 観測頻度を重みとして扱う
        raw_estimates.append((lambda_est, weight))

    # もし固有値が近い（誤差以内）場合にまとめたければクラスタリングする
    if phase_clustering_tol is not None and phase_clustering_tol > 0:
        # もっと本格的なクラスタリング（k-means等）もあり得るが，ここでは単純にソート→隣接要素をマージ
        raw_estimates.sort(key=lambda x: x[0])  # λでソート
        merged_estimates = []
        cur_val, cur_w = raw_estimates[0]
        for i in range(1, len(raw_estimates)):
            nxt_val, nxt_w = raw_estimates[i]
            if abs(nxt_val - cur_val) < phase_clustering_tol:
                # 近いのでまとめる（λ は重み付き平均にしてもよいが簡単のため片方を使う）
                new_val = (cur_val * cur_w + nxt_val * nxt_w) / (cur_w + nxt_w)
                new_w = cur_w + nxt_w
                cur_val, cur_w = new_val, new_w
            else:
                merged_estimates.append((cur_val, cur_w))
                cur_val, cur_w = nxt_val, nxt_w
        merged_estimates.append((cur_val, cur_w))
        # ソート済みを返す
        eigenvalues_list = merged_estimates
    else:
        # ソートして返す
        eigenvalues_list = sorted(raw_estimates, key=lambda x: x[0])

    return eigenvalues_list


def perform_qpca_qpe_blockencoding_all_eigs(scaled_data, simulator, config):
    """
    QPEで得られた測定結果の分布をすべて解析し，複数の固有値とその寄与（確率）を推定する版。
    """
    n_features = scaled_data.shape[1]
    n_qubits = int(np.log2(n_features))
    if 2**n_qubits != n_features:
        raise ValueError("特徴量数が 2^n である必要があります。")

    # --- 1. 平均密度行列 rho を作る ---
    rho = np.zeros((n_features, n_features), dtype=complex)
    qubits_for_data = [cirq.LineQubit(i) for i in range(n_qubits)]

    for sample in scaled_data:
        circuit = create_quantum_encoding_circuit(
            qubits_for_data,
            sample,
            method=config.get('encoding_method', 'amplitude')
        )
        result = simulator.simulate(circuit)
        state = result.final_state_vector
        dm = np.outer(state, np.conjugate(state))
        rho += dm

    rho /= len(scaled_data)

    # --- 2. QPE 実行 ---
    t = config.get("time_parameter", 1.0)
    num_ancilla = config.get("num_ancilla", 3)
    qpe_circuit, ancilla_qubits, data_qubits = create_qpe_circuit_for_rho(
        rho, t, num_ancilla)

    if config.get('print_circuit', False):
        print("=== QPE Circuit (block-encoding) ===")
        print(qpe_circuit)

    repetitions = config.get("repetitions", 1000)
    result = simulator.run(qpe_circuit, repetitions=repetitions)

    # --- 3. ビット列の分布を集計 ---
    counts = {}
    for r in range(repetitions):
        bits = [int(result.measurements[f"m{i}"][r])
                for i in range(num_ancilla)]
        bit_val = 0
        for bit in bits:
            bit_val = (bit_val << 1) | bit
        counts[bit_val] = counts.get(bit_val, 0) + 1

    # --- 4. 測定分布から固有値と確率を推定 ---
    #     phase_clustering_tol は必要に応じて指定
    phase_clustering_tol = config.get("phase_clustering_tol", None)
    eigenvalues_list = extract_all_eigenvalues_from_qpe_measurements(
        counts, num_ancilla, t, repetitions, phase_clustering_tol
    )

    # eigenvalues_list は [(lambda_est, prob), ... ] になっている
    # trace(rho) = 1 を仮定すると，各固有値 λ_i の寄与率は prob_i に相当
    # （※ QPE 誤差や繰り返し数不足により多少のずれはあります）
    # 順序を大きい固有値順に並べ直してみる
    eigenvalues_list_sorted = sorted(
        eigenvalues_list, key=lambda x: x[0], reverse=True)

    # --- 5. 結果の表示 ---
    if config.get('print_eigenvalues', False):
        print("=== QPE Measurement Distribution (All Eigenvalues) ===")
        for idx, (lam, p) in enumerate(eigenvalues_list_sorted):
            print(f"Est EigVal #{idx+1}: {lam:.4f}  (prob ~ {p:.4f})")

    # --- 6. （参考）古典的に rho を固有値分解し比較する ---
    if config.get('compare_with_classical', False):
        classical_eigvals, _ = eigh(rho)
        classical_eigvals_sorted = np.sort(classical_eigvals)[::-1]
        print("\n[Compare with classical diagonalization]")
        for i, val in enumerate(classical_eigvals_sorted):
            print(f"Classical EigVal #{i+1}: {val:.4f}")

    return eigenvalues_list_sorted, rho


def calculate_contribution_ratios_from_qpe(eigenvalues_list):
    """
    QPEで推定した (lambda_est, probability) のリストから，
    固有値と寄与率テーブルを返す。
    """
    # 念のため確率の合計で正規化しておく
    total_prob = sum([p for (_, p) in eigenvalues_list])
    if abs(total_prob) < 1e-10:
        print("Warning: total probability ~ 0. QPE shots might be too few.")
        total_prob = 1e-10

    # 大きい順に並べる
    sorted_list = sorted(eigenvalues_list, key=lambda x: x[0], reverse=True)

    data = []
    for i, (lam, p) in enumerate(sorted_list):
        # 規格化した重み
        norm_prob = p / total_prob
        data.append({
            'Component': f'Eig#{i+1}',
            'Eigenvalue': lam,
            'Probability(approx λ)': p,       # 生の頻度
            'Normalized Contribution': norm_prob,
        })

    df = pd.DataFrame(data)
    return df


# =============================================================================
# 5. （後半コード）簡易ユニタリを用いた qPCA + QPE
# =============================================================================


def get_unitary_from_density_matrix(density_matrix):
    """
    密度行列から「単一量子ビットの回転ゲート」としてユニタリを構成する（非常に簡易的な例）。
    """
    eigenvalues, eigenvectors = eigh(density_matrix)
    # 最大固有値（principal component）だけを見て回転角を決める
    principal_eigenvalue = eigenvalues[-1]
    # 実際は多ビットの場合は行列サイズ分のユニタリが必要になるが，ここではサンプルとして1量子ビット例
    theta = principal_eigenvalue * 2 * np.pi
    unitary = cirq.ry(theta)
    return unitary


def create_qpe_circuit_simple(qubits, unitary, ancilla_qubits, num_ancilla):
    """
    簡易的な1量子ビットユニタリを制御してQPEを行う例。
    qubits: データ用量子ビット（1量子ビット想定）
    unitary: 単一ビットのユニタリ（cirq.Gate）
    ancilla_qubits: QPE用の補助量子ビット
    num_ancilla: 補助量子ビットの数
    """
    circuit = cirq.Circuit()
    # 初期化（補助量子ビットを|+>に）
    circuit.append([cirq.H(q) for q in ancilla_qubits])

    # ユニタリを 2^i 回適用 (制御付き)
    for i in range(num_ancilla):
        exponent = 2**i
        controlled_unitary = cirq.ControlledGate(unitary ** exponent)
        # qubits[0] に作用させる
        circuit.append(controlled_unitary.on(ancilla_qubits[i], qubits[0]))

    # 逆QFT
    circuit.append(cirq.inverse(cirq.qft(*ancilla_qubits)))

    # 測定
    circuit.append([cirq.measure(q, key=f'm{idx}')
                   for idx, q in enumerate(ancilla_qubits)])
    return circuit


def perform_qpca_qpe_simple(circuits, num_iterations, simulator, config):
    """
    (後半コードの方式)
    - 各サンプル回路をシミュレートして density matrix を作り，
    - その平均を取り，最大固有値に対応する 1量子ビット回転ゲートを作成
    - QPEで位相推定（あくまで1量子ビットユニタリの例）
    """
    density_matrices = []
    for circuit in circuits:
        result = simulator.simulate(circuit)
        state = result.final_state_vector
        dm = np.outer(state, np.conj(state))
        density_matrices.append(dm)

    avg_density_matrix = np.mean(density_matrices, axis=0)

    # 単一量子ビットのユニタリを取得
    unitary = get_unitary_from_density_matrix(avg_density_matrix)

    num_ancilla = num_iterations
    ancilla_qubits = [cirq.LineQubit(i) for i in range(num_ancilla)]
    data_qubit = cirq.LineQubit(num_ancilla)  # 1量子ビット想定

    qpe_circuit = create_qpe_circuit_simple(
        [data_qubit], unitary, ancilla_qubits, num_ancilla)

    if config.get('print_circuit', False):
        print("=== QPE Circuit (simple) ===")
        print(qpe_circuit)

    # シミュレーション
    # repetitions=1 だと統計が出ないので，必要なら増やす
    result = simulator.run(qpe_circuit, repetitions=1)

    # 測定結果から位相推定
    eigenvalue_bits = [result.measurements[f'm{
        idx}'][0][0] for idx in range(num_ancilla)]
    eigenvalue = 0
    for bit in eigenvalue_bits:
        eigenvalue = (eigenvalue << 1) | bit
    estimated_eigenvalue = eigenvalue / (2**num_ancilla)

    if config.get('print_eigenvalues', False):
        print("=== QPE (simple) Estimated Eigenvalue ===")
        print(estimated_eigenvalue)

    # クラシカルに平均密度行列を固有値分解
    eigenvalues, eigenvectors = eigh(avg_density_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvalues = eigenvalues[sorted_indices[:num_iterations]]
    top_eigenvectors = eigenvectors[:, sorted_indices[:num_iterations]]

    return top_eigenvectors, top_eigenvalues


# =============================================================================
# 6. 古典PCA & 可視化関連
# =============================================================================

def perform_classical_pca(scaled_data, num_components=2, config=None):
    """
    古典的なPCAを実施し、次元削減などを行う。
    """
    pca = PCA(n_components=num_components)
    transformed_data = pca.fit_transform(scaled_data)
    eigenvalues = pca.explained_variance_
    components = pca.components_
    contribution_ratios = pca.explained_variance_ratio_

    if config and config.get('print_classical_pca', False):
        print("Classical PCA Eigenvalues:", eigenvalues)
        print("Classical PCA Components:", components)
        print("Classical PCA Contribution Ratios:", contribution_ratios)

    return transformed_data, eigenvalues, components, contribution_ratios


def decode_quantum_data(top_eigenvectors, scaled_data):
    """
    qPCAで得た固有ベクトルを用いて古典データを低次元に射影。
    （簡易版: 実数部分だけ使う）
    """
    top_eigenvectors_reduced = top_eigenvectors[:scaled_data.shape[1], :].T.real
    reduced_data = np.dot(scaled_data, top_eigenvectors_reduced)
    return reduced_data


def calculate_contribution_ratios(eigenvalues, num_components):
    """
    固有値の寄与率を計算。
    """
    total_variance = np.sum(eigenvalues)
    num_available = len(eigenvalues)
    num_selected = min(num_components, num_available)

    if num_selected < num_components:
        print(f"Warning: Requested num_components={num_components} exceeds available eigenvalues={
              num_available}. Adjusting to {num_selected}.")

    selected_eigenvalues = eigenvalues[:num_selected]
    contribution_ratios = selected_eigenvalues / total_variance

    contribution_table = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(num_selected)],
        'Eigenvalue': selected_eigenvalues,
        'Contribution Ratio': contribution_ratios
    })

    return contribution_table


# -------------------------------
# 以下は各種プロット関数 (省略可)
# -------------------------------
def plot_original_data(scaled_data, labels, dataset_name, config):
    """
    元データを2次元で可視化。
    """
    plot_config = config.get('plotting', {})
    if not plot_config.get('plot_original_data', False):
        return

    save_fig = plot_config.get('save_figures', False)
    fig_path = plot_config.get('original_data_fig_path', f"{
                               dataset_name}_original_data.png")

    # 次元が2より大きい場合はPCAで2次元に
    if scaled_data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(scaled_data)
        title = f"{dataset_name} Original Data (PCA Reduced to 2D)"
    else:
        data_2d = scaled_data
        title = f"{dataset_name} Original Data"

    plt.figure(figsize=(8, 6))
    if labels is not None:
        unique_labels = np.unique(labels)
        palette = sns.color_palette("hsv", len(unique_labels))
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1],
                        hue=labels, palette=palette, legend='full')
    else:
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1])
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()

    if save_fig:
        plt.savefig(fig_path)
        print(f"Original data plot saved to {fig_path}")
    else:
        plt.show()
    plt.close()


def plot_qpca_results(decoded_data, labels, dataset_name, config):
    """
    qPCAで次元削減した結果を可視化。
    """
    plot_config = config.get('plotting', {})
    if not plot_config.get('plot_qpca_results', False):
        return

    save_fig = plot_config.get('save_figures', False)
    fig_path = plot_config.get('qpca_results_fig_path', f"{
                               dataset_name}_qpca_results.png")

    if decoded_data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(decoded_data)
        title = f"{dataset_name} qPCA Results (PCA Reduced to 2D)"
    else:
        data_2d = decoded_data
        title = f"{dataset_name} qPCA Results"

    plt.figure(figsize=(8, 6))
    if labels is not None:
        unique_labels = np.unique(labels)
        palette = sns.color_palette("hsv", len(unique_labels))
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1],
                        hue=labels, palette=palette, legend='full')
    else:
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1])
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()

    if save_fig:
        plt.savefig(fig_path)
        print(f"qPCA results plot saved to {fig_path}")
    else:
        plt.show()
    plt.close()


def plot_classical_pca_results(classical_pca_data, labels, dataset_name, config):
    """
    古典PCAで次元削減した結果を可視化。
    """
    plot_config = config.get('plotting', {})
    if not plot_config.get('plot_classical_pca_results', False):
        return

    save_fig = plot_config.get('save_figures', False)
    fig_path = plot_config.get('classical_pca_results_fig_path', f"{
                               dataset_name}_classical_pca_results.png")

    if classical_pca_data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(classical_pca_data)
        title = f"{dataset_name} Classical PCA Results (PCA Reduced to 2D)"
    else:
        data_2d = classical_pca_data
        title = f"{dataset_name} Classical PCA Results"

    plt.figure(figsize=(8, 6))
    if labels is not None:
        unique_labels = np.unique(labels)
        palette = sns.color_palette("hsv", len(unique_labels))
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1],
                        hue=labels, palette=palette, legend='full')
    else:
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1])
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()

    if save_fig:
        plt.savefig(fig_path)
        print(f"Classical PCA results plot saved to {fig_path}")
    else:
        plt.show()
    plt.close()


def plot_eigenvalues(eigenvalues, dataset_name, config, method='qPCA'):
    """
    固有値をバーで可視化。method='qPCA' or 'Classical PCA'
    """
    plot_config = config.get('plotting', {})
    if not plot_config.get('plot_eigenvalues', False):
        return

    save_fig = plot_config.get('save_figures', False)
    fig_path = plot_config.get('eigenvalues_fig_path', f"{
                               dataset_name}_eigenvalues_{method}.png")

    num_components = len(eigenvalues)
    sorted_eigenvalues = eigenvalues

    plt.figure(figsize=(8, 6))
    sns.barplot(x=np.arange(1, num_components + 1),
                y=sorted_eigenvalues, color="skyblue")
    plt.title(f"{dataset_name} {method} Eigenvalues (Sorted)")
    plt.xlabel("Component")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()

    if save_fig:
        plt.savefig(fig_path)
        print(f"{method} Eigenvalues plot saved to {fig_path}")
    else:
        plt.show()
    plt.close()


def plot_pca_comparison(qpca_eigenvalues,
                        classical_eigenvalues,
                        qpca_contribution,
                        classical_contribution,
                        dataset_name, config):
    """
    qPCAと古典PCAの固有値＆寄与率を比較してプロット。
    """
    plot_config = config.get('plotting', {})
    if not plot_config.get('plot_pca_comparison', False):
        return

    save_fig = plot_config.get('save_figures', False)
    fig_path_eigen = plot_config.get('pca_comparison_eigenvalues_fig_path', f"{
                                     dataset_name}_pca_comparison_eigenvalues.png")
    fig_path_contribution = plot_config.get('pca_comparison_contribution_fig_path', f"{
                                            dataset_name}_pca_comparison_contribution.png")

    num_components = len(qpca_eigenvalues)
    indices = np.arange(num_components)
    width = 0.35

    # 1. 固有値比較
    plt.figure(figsize=(10, 6))
    plt.bar(indices - width/2, qpca_eigenvalues,
            width=width, label='qPCA', color='skyblue')
    plt.bar(indices + width/2, classical_eigenvalues,
            width=width, label='Classical PCA', color='salmon')
    plt.title(f"{dataset_name} PCA Comparison of Eigenvalues")
    plt.xlabel("Component")
    plt.ylabel("Eigenvalue")
    plt.xticks(indices, [f'PC{i+1}' for i in range(num_components)])
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig(fig_path_eigen)
        print(f"PCA comparison Eigenvalues plot saved to {fig_path_eigen}")
    else:
        plt.show()
    plt.close()

    # 2. 寄与率比較
    plt.figure(figsize=(10, 6))
    plt.bar(indices - width/2, qpca_contribution,
            width=width, label='qPCA', color='skyblue')
    plt.bar(indices + width/2, classical_contribution,
            width=width, label='Classical PCA', color='salmon')
    plt.title(f"{dataset_name} PCA Comparison of Contribution Ratios")
    plt.xlabel("Component")
    plt.ylabel("Contribution Ratio")
    plt.xticks(indices, [f'PC{i+1}' for i in range(num_components)])
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig(fig_path_contribution)
        print(f"PCA comparison Contribution Ratios plot saved to {
              fig_path_contribution}")
    else:
        plt.show()
    plt.close()


# =============================================================================
# 7. メイン関数
# =============================================================================

def main():
    # 1. 設定ファイル読み込み
    config = load_config(CONFIG_PATH)

    # 2. シード設定
    seed = set_random_seed(config)
    if seed is not None:
        simulator = cirq.Simulator(seed=seed)
    else:
        simulator = cirq.Simulator()

    # configから取得
    dataset_names = config.get('dataset_name', 'iris').split(',')
    normalization_methods = config.get(
        'normalization_method', 'standard').split(',')
    encoding_methods = config.get('encoding_method', 'amplitude').split(',')
    desired_num_components = config.get('num_components', 2)
    qpca_mode = config.get('qpca_mode', 'block')   # 'block' or 'simple' など

    # 3. 各データセット × 各正規化 × 各エンコード手法 に対し処理
    for dname in dataset_names:
        dname = dname.strip()
        print(f"\n=== Dataset: {dname} ===")
        for norm_method in normalization_methods:
            norm_method = norm_method.strip()
            print(f"Normalization: {norm_method}")
            scaled_data, labels = load_and_scale_dataset(
                dname, norm_method, random_state=seed if seed is not None else None)

            # 4. モードに応じて「ブロックエンコード版 qPCA」or「簡易版 qPCA」などを実行
            if qpca_mode == 'block':
                # print("Mode: qPCA (block-encoding)")
                # ----- ブロックエンコード方式 -----
                classical_eigvals_sorted, lambda_est, rho = perform_qpca_qpe_blockencoding(
                    scaled_data, simulator, config
                )
                # ここでは主に固有値を取得（固有ベクトルはブロックエンコード方式では大きい行列）
                top_eigenvalues = classical_eigvals_sorted[:desired_num_components]

                # クラシカルPCAや可視化をしたい場合は，
                # qPCAによる「データ射影」をどう実装するか別途検討が必要。
                # （ブロックエンコード方式だと固有ベクトルの抽出が大規模行列になるため簡易実装は省略）

                # 一応固有値の簡易プロット
                if config.get('plotting', {}).get('plot_eigenvalues', False):
                    plot_eigenvalues(
                        top_eigenvalues,
                        dname, config, method='qPCA(block-encoding)'
                    )

            # 量子だけで全固有値を求める！
            elif qpca_mode == 'block_all':
                eigenvalues_list, rho = perform_qpca_qpe_blockencoding_all_eigs(
                    scaled_data, simulator, config
                )

                # 寄与率を計算
                contrib_df = calculate_contribution_ratios_from_qpe(
                    eigenvalues_list)
                print("\nQPE (All Eigs) Contribution Ratios Table:")
                print(contrib_df)

            elif qpca_mode == 'simple':
                # ----- 簡易1量子ビットユニタリ方式 -----
                # データをエンコードする回路群を作成
                # （サンプルの数だけ回路を作り，追加でノイズを混ぜたりなど）
                circuits = []
                for data_point in scaled_data:
                    circuit = create_quantum_encoding_circuit(
                        [cirq.LineQubit(0)],  # ここでは1量子ビットに無理やりエンコード
                        data_point,
                        method=encoding_methods[0]  # 複数指定時は先頭を使う例
                    )
                    circuits.append(circuit)

                top_eigenvectors, top_eigenvalues = perform_qpca_qpe_simple(
                    circuits, desired_num_components, simulator, config
                )

                # 復元（デコード）したデータ（= 低次元空間）
                decoded_data = decode_quantum_data(
                    top_eigenvectors, scaled_data)
                print(f"Decoded data shape: {decoded_data.shape}")

                # 可視化
                if config.get('plotting', {}).get('plot_qpca_results', False):
                    plot_qpca_results(decoded_data, labels, dname, config)

                # 寄与率計算
                contribution_table = calculate_contribution_ratios(
                    top_eigenvalues, desired_num_components)
                print("qPCA (simple) Contribution Ratios Table:")
                print(contribution_table)

            else:
                print(f"[Warning] Unknown qpca_mode: {qpca_mode}")

            # 5. 古典PCA
            perform_classical_pca_flag = config.get(
                'perform_classical_pca', False)
            if perform_classical_pca_flag:
                classical_pca_data, classical_eigenvalues, classical_components, classical_contribution_ratios = perform_classical_pca(
                    scaled_data, num_components=desired_num_components, config=config
                )

                # 可視化
                if config.get('plotting', {}).get('plot_classical_pca_results', False):
                    plot_classical_pca_results(
                        classical_pca_data, labels, dname, config)

                # 寄与率テーブル
                classical_contribution_table = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(classical_eigenvalues))],
                    'Eigenvalue': classical_eigenvalues,
                    'Contribution Ratio': classical_contribution_ratios
                })
                print("Classical PCA Contribution Ratios Table:")
                print(classical_contribution_table)

                # qPCAと古典PCAの比較グラフ
                #   - ブロックエンコード版の場合は固有値を top_eigenvalues として使うかは要検討
                #   - 簡易版の場合は top_eigenvalues があるので比較OK
                if config.get('plotting', {}).get('plot_pca_comparison', False) and qpca_mode == 'simple':
                    # 上位成分のみ
                    num_qpca_eig = len(top_eigenvalues)
                    num_classical_eig = len(classical_eigenvalues)
                    limit = min(num_qpca_eig, num_classical_eig)
                    plot_pca_comparison(
                        top_eigenvalues[:limit],
                        classical_eigenvalues[:limit],
                        contribution_table['Contribution Ratio'].values[:limit],
                        classical_contribution_ratios[:limit],
                        dname, config
                    )

            # 6. 元データの可視化（2次元など）
            if config.get('plotting', {}).get('plot_original_data', False):
                plot_original_data(scaled_data, labels, dname, config)


if __name__ == "__main__":
    main()

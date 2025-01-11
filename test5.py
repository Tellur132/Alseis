import cirq
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def classical_pca_iris(n_components=2):
    """
    Irisデータセットに対して古典的PCAを行い，主成分を返す．
    """
    iris = load_iris()
    X = iris.data  # (150, 4)
    # 標準化（平均0，分散1）
    X_std = StandardScaler().fit_transform(X)

    # 古典的PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)

    # 固有値・固有ベクトル
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_

    return X_pca, eigenvalues, eigenvectors, explained_variance_ratio, iris


def encode_data_into_quantum_state(data_point, qubits):
    """
    1サンプルの特徴量を量子振幅に埋め込む例（簡易的）．
    data_point はすでに正規化済(ノルム1近傍)だと仮定．
    """
    circuit = cirq.Circuit()

    # 振幅エンコードでは本来4つの要素をそのまま振幅に載せる必要があるので
    # 4量子ビットが本格的には必要
    # ここでは便宜的に2量子ビットに集約し，回転角 + 位相で表現するイメージの例

    # データを0~1に正規化して，それを回転角にするイメージ
    # （注意：これはガチの振幅エンコードではありません）
    theta1 = data_point[0] * np.pi
    theta2 = data_point[1] * np.pi
    circuit.append(cirq.ry(theta1)(qubits[0]))
    circuit.append(cirq.ry(theta2)(qubits[1]))

    # 他の成分は位相として適当に反映(例)
    phi1 = data_point[2] * np.pi
    phi2 = data_point[3] * np.pi
    circuit.append(cirq.rz(phi1)(qubits[0]))
    circuit.append(cirq.rz(phi2)(qubits[1]))

    return circuit


def construct_qpe_circuit(rho_qubits, ancilla_qubits, unitary):
    """
    量子位相推定(QPE)回路を構築する．
    - rho_qubits: データをエンコードした量子ビット
    - ancilla_qubits: 位相推定用アシリア量子ビット
    - unitary: 推定したいユニタリゲート（例: cirq.ry(angle)）
    """
    circuit = cirq.Circuit()

    # Step 1: アシリアビットを |0> + |1> のスーパーで初期化
    for aq in ancilla_qubits:
        circuit.append(cirq.H(aq))

    # Step 2: 制御ユニタリの適用
    # QPEでは位相を2^k倍にして得るので，アシリアビットに応じて何回もユニタリをかける
    for i, aq in enumerate(ancilla_qubits):
        # 2^i 回ユニタリをかける
        exponent = 2**i
        for _ in range(exponent):
            # rho_qubits の各量子ビットに対して制御ユニタリを適用
            for rho_qubit in rho_qubits:
                circuit.append(unitary.controlled().on(aq, rho_qubit))

    # Step 3: アシリアビットに逆フーリエ変換 (QFT†) を適用
    # QPEの定石
    circuit = apply_inverse_qft(circuit, ancilla_qubits)

    # 計測
    circuit.append(cirq.measure(*ancilla_qubits, key='phase_est'))

    return circuit


def apply_inverse_qft(circuit, qubits):
    """
    QFT^-1 をアシリアビットに適用する
    （簡易実装; 実際はたくさんの制御回転ゲートが必要）
    """
    # 一般的なQFT^-1の実装（全ビット分）例
    # 公式実装もあるが，ここでは手作りしてみる
    n = len(qubits)
    for i in range(n//2):
        circuit.append(cirq.SWAP(qubits[i], qubits[n-1-i]))

    for i in range(n):
        for j in range(i):
            circuit.append(cirq.CZ(qubits[i], qubits[j])**(-1/2**(i-j)))
        circuit.append(cirq.H(qubits[i]))

    return circuit


def qPCA_iris(num_samples=10, t=1.0, num_eigenvalues=2):
    """
    Irisデータをいくつかサンプリングして，qPCAを試みる関数．
    - num_samples: 量子状態に埋め込むサンプル数
    - t: rho に対する e^{i * rho * t} の t (本来はアルゴリズム設計による)
    - num_eigenvalues: 推定したい固有値の数
    """
    iris = load_iris()
    X = iris.data
    X_std = StandardScaler().fit_transform(X)

    # サンプルを取る（本来は全データの平均密度行列を作るが，簡易化）
    indices = np.random.choice(len(X_std), size=num_samples, replace=False)
    sampled_data = X_std[indices]

    # 主成分を求めるために、共分散行列を計算
    cov_matrix = np.cov(sampled_data.T)

    # 共分散行列の固有値分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 上位の固有値と固有ベクトルを選択
    top_indices = np.argsort(eigenvalues)[::-1][:num_eigenvalues]
    top_eigenvalues = eigenvalues[top_indices]
    top_eigenvectors = eigenvectors[:, top_indices]

    # 各固有値に対してQPEを行う
    explained_variances_qpca = []
    estimated_eigenvalues = []
    circuits = []
    for i in range(num_eigenvalues):
        eigenvalue = top_eigenvalues[i]
        eigenvector = top_eigenvectors[:, i]

        # ユニタリ演算子 U = e^{i * eigenvalue * t}
        # ここでは、単純に Ry 回転ゲートを使用して近似
        unitary_op = cirq.ry(eigenvalue * t)

        # 量子ビットの準備
        rho_qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
        ancilla_qubits = [cirq.LineQubit(
            2 + 2*i), cirq.LineQubit(3 + 2*i)]  # 各固有値ごとにアシリアビットを用意

        # 回路構築
        circuit = cirq.Circuit()
        # 固有ベクトルをエンコード
        encoded_circuit = encode_data_into_quantum_state(
            eigenvector, rho_qubits)
        circuit += encoded_circuit

        # QPE 回路を構築
        qpe_circuit = construct_qpe_circuit(
            rho_qubits=rho_qubits,
            ancilla_qubits=ancilla_qubits,
            unitary=unitary_op  # 修正箇所: 単一のゲートを直接渡す
        )
        circuit += qpe_circuit

        # シミュレータ
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)

        # 位相推定結果
        phase_measurements = result.measurements['phase_est']
        # アシリアビットの数に応じて位相を計算
        num_ancilla = len(ancilla_qubits)
        decimal_phases = np.zeros(len(phase_measurements))
        for bit in range(num_ancilla):
            decimal_phases += phase_measurements[:, bit] * (2 ** bit)
        estimated_phase = np.mean(decimal_phases) / (2**num_ancilla)

        # 位相から近似固有値を計算
        estimated_eigenvalue = estimated_phase / t

        # 寄与率を計算
        explained_variance_qpca = estimated_eigenvalue / \
            np.sum(top_eigenvalues)

        explained_variances_qpca.append(explained_variance_qpca)
        estimated_eigenvalues.append(estimated_eigenvalue)
        circuits.append(circuit)

    return circuits, explained_variances_qpca, estimated_eigenvalues, top_eigenvalues


def plot_pca(X_pca, iris, title="PCA"):
    """
    PCA結果を2次元プロットする関数
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.colorbar(scatter, label='Species')
    plt.grid(True)
    plt.show()


def main():
    # 1. 古典PCA
    X_pca, eigenvalues, eigenvectors, explained_variance_ratio, iris = classical_pca_iris()
    print("=== Classical PCA ===")
    print("Explained Variance (eigenvalues):", eigenvalues)
    print("Explained Variance Ratio:", explained_variance_ratio)
    print("Principal Components (first 2):\n", eigenvectors[:2])
    print()

    # 2. 真の qPCA に近い実装
    circuits, explained_variances_qpca, estimated_eigenvalues, true_eigenvalues = qPCA_iris(
        num_samples=10, t=1.0, num_eigenvalues=2)
    print("=== Quantum PCA (like) ===")
    for i in range(len(circuits)):
        print(f"--- Eigenvalue {i+1} ---")
        print("Constructed QPE circuit:\n", circuits[i])
        print(f"Estimated Eigenvalue {
              i+1} from QPE:", estimated_eigenvalues[i])
        print(f"True Eigenvalue {i+1}:", true_eigenvalues[i])
        print(f"Estimated Explained Variance (qPCA) {
              i+1}:", explained_variances_qpca[i])
        print()

    # 3. 寄与率の比較
    print("=== Comparison of Explained Variance ===")
    print(f"Classical PCA Explained Variance Ratios: {
          explained_variance_ratio[:2]}")
    print(f"Quantum PCA Estimated Explained Variance Ratios: {
          explained_variances_qpca}")
    print()

    # 4. PCA結果のプロット
    plot_pca(X_pca, iris, title="Classical PCA on Iris Dataset")

    # 5. 量子PCAの結果を表示
    for i, ev in enumerate(explained_variances_qpca):
        print(f"Quantum PCA Explained Variance Ratio {i+1}: {ev:.4f}")


if __name__ == "__main__":
    main()

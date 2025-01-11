import numpy as np
import cirq
import sympy
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

##############################################################################
# 1. QRAC (2ビット → 1量子ビット) のユーティリティ関数
##############################################################################


def create_qrac_state_2to1(qubit: cirq.LineQubit, b1: int, b2: int):
    """
    2つの古典ビット (b1, b2) を 1量子ビットにエンコードする
    2-to-1 QRAC 回路を返す。

    状態:
      (b1,b2) に応じて Bloch 球の赤道上で90°刻みの位置に配置。
      θ = (2*b1 + b2) * (π/2)

    生成される状態 |ψ_{b1,b2}> は
      (|0> + e^{iθ}|1>) / √2
    となる。
    """
    circuit = cirq.Circuit()
    # 初期状態 |0>

    # Step1: Hゲートで |+> = (|0> + |1>)/sqrt{2} を生成
    circuit.append(cirq.H(qubit))

    # Step2: Rz(θ)
    # θ は 4パターン: 0, π/2, π, 3π/2
    theta = (2 * b1 + b2) * (np.pi / 2)
    circuit.append(cirq.rz(theta)(qubit))

    return circuit

##############################################################################
# 2. データのエンコード回路を作る
##############################################################################


def encode_data_qrac(circuit: cirq.Circuit, qubits, data):
    """
    与えられたデータ (4次元) を 2bit×2bit の形でエンコードする例。
    それぞれの2bitを1量子ビットに QRAC でエンコードするので、
    4次元 → (2bit + 2bit) → 2量子ビット とする。

    データは [-∞,∞] の実数ですが、とりあえず簡単のために
    '0より大きいかどうか' でビット化している（サンプル）。
    """
    # data: shape (4,) の想定 (Irisの1サンプル)

    # 量子ビットは2つ使う想定
    qubit0, qubit1 = qubits

    # 0,1ビット目
    bit1 = int(data[0] > 0)  # この閾値化はあくまでサンプル
    bit2 = int(data[1] > 0)
    circuit += create_qrac_state_2to1(qubit0, bit1, bit2)

    # 2,3ビット目
    bit3 = int(data[2] > 0)
    bit4 = int(data[3] > 0)
    circuit += create_qrac_state_2to1(qubit1, bit3, bit4)

    return circuit


##############################################################################
# 3. VQE 用のクラス (簡単版)
##############################################################################
class SimpleVQE:
    def __init__(self, qubits, ansatz_depth=1):
        """
        ansatz_depth: いくつかの回転ゲートを並べる深さ
        """
        self.qubits = qubits
        self.ansatz_depth = ansatz_depth

        # シンボル (変分パラメータ) の用意
        # 量子ビットごと + 層ごとに3つの回転軸を用意 (RX, RY, RZ など)
        self.params = []
        for layer in range(ansatz_depth):
            for q in qubits:
                self.params.append(sympy.Symbol(f'theta_{layer}_{q}_rx'))
                self.params.append(sympy.Symbol(f'theta_{layer}_{q}_ry'))
                self.params.append(sympy.Symbol(f'theta_{layer}_{q}_rz'))
        self.param_symbols = self.params

        # ansatz 回路を定義 (構造だけ)
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self):
        """
        パラメータ化 ansatz 回路を作る
        """
        circuit = cirq.Circuit()

        idx = 0
        for layer in range(self.ansatz_depth):
            for q in self.qubits:
                circuit.append(cirq.rx(self.params[idx])(q))
                idx += 1
                circuit.append(cirq.ry(self.params[idx])(q))
                idx += 1
                circuit.append(cirq.rz(self.params[idx])(q))
                idx += 1

            # 簡単のため，各層の最後に隣接2量子ビットにCZゲート
            # (2量子ビットなのであまり意味は大きくないがデモ)
            for i in range(len(self.qubits)-1):
                circuit.append(cirq.CZ(self.qubits[i], self.qubits[i+1]))

        return circuit

    def expectation(self, circuit: cirq.Circuit, operator: cirq.PauliSum, param_values):
        """
        与えられた circuit に自分の ansatz を差し込み，operator の期待値を計算
        """
        # パラメータを実際の値に置き換えて circuit を最適化
        resolver = dict(zip(self.param_symbols, param_values))
        full_circuit = cirq.Circuit()
        full_circuit += circuit  # データエンコード部分
        full_circuit += self.ansatz  # ansatz (変分回路)

        # シミュレーション
        simulator = cirq.Simulator()
        # 拡張: cirqPauliSumCollector を使う方法もあるが，ここは簡単に求める
        result = simulator.simulate(full_circuit, param_resolver=resolver)
        final_state = result.final_state_vector

        # 期待値計算
        # 演算子 operator は例えば cirq.Z(qubit0)*cirq.Z(qubit1) + ... みたいな和
        exp_val = 0
        for p in operator:
            # p: cirq.PauliString
            m = cirq.to_valid_state_vector(final_state, len(self.qubits))
            # cirq.expectationを使う場合 (PauliString 1つ限定)
            # ここでは cirq.PauliString の expectation_from_state_vector を用いる
            expect_val = operator.expectation_from_state_vector(
                state_vector=final_state,
                qubit_map={q: i for i, q in enumerate(self.qubits)}
            )
        return exp_val

    def minimize(self, circuit: cirq.Circuit, operator: cirq.PauliSum, init_guess=None,
                 maxiter=50, lr=0.1):
        """
        operator の期待値を最小化するようにパラメータを学習する (簡単なGD)
        """
        if init_guess is None:
            init_guess = np.random.uniform(-np.pi, np.pi, len(self.params))

        param_values = init_guess

        for step in range(maxiter):
            # 数値微分で勾配を求める (シンプル実装)
            grad = np.zeros(len(param_values), dtype=float)

            base_val = self.expectation(circuit, operator, param_values)
            for i in range(len(param_values)):
                shift = 1e-3
                param_values[i] += shift
                plus_val = self.expectation(circuit, operator, param_values)
                param_values[i] -= 2*shift
                minus_val = self.expectation(circuit, operator, param_values)
                param_values[i] += shift
                grad[i] = (plus_val - minus_val) / (2*shift)

            # パラメータ更新 (gradient descent)
            param_values -= lr * grad

            # 収束チェック（ざっくり）
            if step % 10 == 0:
                print(f"[step {step}] expectation = {base_val:.4f}")

        final_val = self.expectation(circuit, operator, param_values)
        return final_val, param_values


##############################################################################
# 4. メイン処理: Irisデータに対してエンコード & VQE (固有値探索もどき)
##############################################################################
def main():
    # ---------------------------
    # 4-1. Irisデータセット読み込み
    # ---------------------------
    iris = load_iris()
    X = iris.data  # (150, 4)
    y = iris.target

    # データの標準化
    X = StandardScaler().fit_transform(X)

    # ---------------------------
    # 4-2. 古典的PCA (2次元) を実行
    # ---------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)  # shape (150, 2)
    # これは後で可視化用

    # ---------------------------
    # 4-3. 量子ビット準備
    # ---------------------------
    # ここでは2量子ビットで表現 (2-to-1 QRAC を2回使う)
    qubits = [cirq.LineQubit(i) for i in range(2)]

    # ---------------------------
    # 4-4. 量子「相関行列演算子」(の簡易バージョン) の定義
    # ---------------------------
    # 本来のqPCAだと、データから密度行列や相関行列を構成し、
    # それを対角化するためのユニタリを探す…という流れになる。
    #
    # ここではお試しとして「Z演算子同士の相関」をとるような演算子を
    # 適当に組んで VQE で固有値を探索する流れにする。
    #
    # 例: H = a*Z0 + b*Z1 + c*(Z0*Z1)
    # という PauliSum をデータごとに求めるイメージ
    # (あくまでデモであり，ちゃんと相関行列を表しているわけではない)

    def construct_operator_from_data(sample, qubits):
        """
        サンプルに応じて演算子 H を作成し、PauliSum として返す。
        例:
        H = sample[0]*Z0 + sample[1]*Z1 + (sample[2]*sample[3])*Z0Z1
        """
        Z0 = cirq.Z(qubits[0])
        Z1 = cirq.Z(qubits[1])

        # PauliString 同士を足し合わせて PauliSum を作る
        # cirq.Z(qubit0)*cirq.Z(qubit1) は PauliString(Z0, Z1) と書ける。
        # それに係数(coefficient)を掛けて足し合わせる場合は、
        #   cirq.PauliString(..., coefficient=○○)
        # のように書くのが安全。
        pZ0 = cirq.PauliString(Z0,   coefficient=sample[0])
        pZ1 = cirq.PauliString(Z1,   coefficient=sample[1])
        pZ0Z1 = cirq.PauliString(Z0, Z1, coefficient=sample[2] * sample[3])

        # 最終的に PauliSum として返す
        H = pZ0 + pZ1 + pZ0Z1

        return H

    # ---------------------------
    # 4-5. サンプルを1つだけ取り出して VQE で固有値探し (デモ)
    # ---------------------------
    # 例として，最初の1サンプルだけ qPCA(もどき) を適用
    sample_data = X[0]  # 1サンプル (4次元)

    # データエンコード回路作成
    circuit = cirq.Circuit()
    circuit = encode_data_qrac(circuit, qubits, sample_data)

    # 演算子 (ハミルトニアン風) を作成
    H = construct_operator_from_data(sample_data, qubits)

    # VQEインスタンス用意
    vqe_solver = SimpleVQE(qubits, ansatz_depth=2)

    print("=== VQEで演算子の最小固有値を探索(っぽいこと)をします ===")
    val, params = vqe_solver.minimize(circuit, H, maxiter=50, lr=0.1)
    print(f"最終的な推定固有値 (最小) : {val:.4f}")

    # （本来qPCAなら最大固有値も探すなどいろいろ行う）

    # ---------------------------
    # 5. 古典PCAとの比較グラフ (2次元)
    # ---------------------------
    # 今回は全サンプルの古典PCA結果 (X_pca) を単に散布図で表示
    # “qPCA”結果としては固有値1つしか計算していないので，比較はあくまで雰囲気
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.title("Classical PCA (Iris) - 2 principal components")
    plt.colorbar(label='Iris class')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

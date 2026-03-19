"""
Decision Tree Educacional

"""

import numpy as np
from collections import Counter
import math


class DecisionTreeNode:
    """Nó da árvore com metadados"""
    def __init__(self, depth=0, node_id=0):
        self.depth = depth
        self.node_id = node_id
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.samples = 0
        self.impurity = None
        self.class_dist = {}
        self.gain = None
        
    def is_leaf(self):
        return self.value is not None


class EducationalDecisionTree:
    """Árvore de decisão"""
    
    def __init__(self, criterion='gini', max_depth=3, min_samples_split=2, verbose_simple=True):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.verbose_simple = verbose_simple if verbose_simple is not None else True
        self.root = None
        self._node_id = 0
        self.logs = []
    
    def _log(self, msg, indent=0, explain=None, level='info'):
        """Adiciona mensagem ao log com explicação"""
        entry = {'msg': "  " * indent + msg, 'type': level}
        if explain and self.verbose_simple:
            entry['explain'] = explain
        self.logs.append(entry)
    
    def _gini(self, y, explain=True):
        """Calcula Gini com explicações"""
        if len(y) == 0:
            return 0.0, []
        counts = Counter(y)
        n = len(y)
        gini = 1.0
        steps = []
        
        if explain:
            if self.verbose_simple:
                steps.append(f"📊 Temos {n} amostras com classes: {dict(counts)}")
                steps.append("💡 O Gini mede o quanto as classes estão 'misturadas' neste grupo.")
            else:
                steps.append(f"• Contagem de classes: {dict(counts)} | n={n}")
        
        for label, cnt in counts.items():
            p = cnt / n
            gini -= p ** 2
            if explain:
                if self.verbose_simple:
                    steps.append(f"  • Classe {label}: {cnt}/{n} = {p:.1%} → contribuição: {p**2:.3f}")
                else:
                    steps.append(f"  p({label})={p:.3f} → p²={p**2:.4f}")
        
        if explain:
            if self.verbose_simple:
                steps.append(f"✅ Gini = 1 - soma das contribuições = {gini:.3f}")
                steps.append(f"💡 Quanto mais próximo de 0, mais 'puro' é o grupo!")
            else:
                steps.append(f"→ Gini = {gini:.4f}")
        
        return gini, steps
    
    def _entropy(self, y, explain=True):
        """Calcula Entropia com explicações"""
        if len(y) == 0:
            return 0.0, []
        counts = Counter(y)
        n = len(y)
        ent = 0.0
        steps = []
        
        if explain:
            if self.verbose_simple:
                steps.append(f"📊 Temos {n} amostras com classes: {dict(counts)}")
                steps.append("💡 Entropia mede a 'desordem': 0 = grupo puro, alto = classes misturadas.")
            else:
                steps.append(f"• Contagem: {dict(counts)} | n={n}")
        
        for label, cnt in counts.items():
            p = cnt / n
            if p > 0:
                term = -p * math.log2(p)
                ent += term
                if explain:
                    if self.verbose_simple:
                        steps.append(f"  • Classe {label}: {p:.1%} → -p·log₂(p) = {term:.3f}")
                    else:
                        steps.append(f"  p={p:.3f} → -p·log₂(p)={term:.4f}")
        
        if explain:
            if self.verbose_simple:
                steps.append(f"✅ Entropia = soma dos termos = {ent:.3f} bits")
                steps.append(f"💡 0 bits = certeza total; valor alto = muita incerteza!")
            else:
                steps.append(f"→ Entropia = {ent:.4f}")
        
        return ent, steps
    
    def _impurity(self, y, explain=True):
        """Função unificada para calcular impureza"""
        if self.criterion == 'gini':
            return self._gini(y, explain)
        return self._entropy(y, explain)
    
    def _information_gain(self, y, y_left, y_right, explain=True):
        """Calcula ganho de informação com narrativa"""
        parent_imp, parent_steps = self._impurity(y, explain)
        n = len(y)
        n_l, n_r = len(y_left), len(y_right)
        
        if n_l == 0 or n_r == 0:
            return -1, parent_steps + (["⚠ Split inválido: um grupo ficou vazio"] if explain else [])
        
        imp_l, _ = self._impurity(y_left, explain=False)
        imp_r, _ = self._impurity(y_right, explain=False)
        weighted = (n_l/n) * imp_l + (n_r/n) * imp_r
        gain = parent_imp - weighted
        
        if explain:
            steps = parent_steps + ["\n🔀 Avaliando este split:"]
            if self.verbose_simple:
                steps.append(f"  • Grupo esquerdo: {n_l} amostras, impureza={imp_l:.3f}")
                steps.append(f"  • Grupo direito: {n_r} amostras, impureza={imp_r:.3f}")
                steps.append(f"  • Impureza média ponderada: ({n_l}/{n})×{imp_l:.3f} + ({n_r}/{n})×{imp_r:.3f} = {weighted:.3f}")
                steps.append(f"✅ Ganho = impureza_antes - impureza_depois = {parent_imp:.3f} - {weighted:.3f} = {gain:.3f}")
                steps.append(f"💡 Ganho alto = split bom! A árvore escolhe o split com MAIOR ganho.")
            else:
                steps.append(f"  Left (n={n_l}): imp={imp_l:.4f} | Right (n={n_r}): imp={imp_r:.4f}")
                steps.append(f"  Weighted: ({n_l}/{n})×{imp_l:.4f} + ({n_r}/{n})×{imp_r:.4f} = {weighted:.4f}")
                steps.append(f"→ Gain = {parent_imp:.4f} - {weighted:.4f} = {gain:.4f}")
        else:
            steps = []
        
        return gain, steps
    
    def _best_split(self, X, y):
        """Encontra o melhor split com logging"""
        best_gain, best_feat, best_thresh = -1, None, None
        best_steps = []
        
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left = X[:, feat] <= thresh
                right = ~left
                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue
                gain, steps = self._information_gain(y, y[left], y[right], explain=False)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
                    best_steps = steps
        
        return best_feat, best_thresh, best_gain, best_steps
    
    def _build(self, X, y, depth=0):
        """Constrói a árvore com explicações"""
        node = DecisionTreeNode(depth=depth, node_id=self._node_id)
        self._node_id += 1
        
        node.samples = len(y)
        node.class_dist = dict(Counter(y))
        node.impurity, imp_steps = self._impurity(y, explain=True)
        
        # Log do nó
        if self.verbose_simple:
            self._log(f"\n🌲 Nó #{node.node_id} (nível {depth})", depth, 
                     "Cada nó representa um grupo de amostras que a árvore está analisando.")
            self._log(f"📦 {node.samples} amostras aqui: {node.class_dist}", depth,
                     "Distribuição mostra quantas amostras de cada classe estão neste grupo.")
        else:
            self._log(f"\n🌲 Node #{node.node_id} (depth {depth})", depth)
            self._log(f"Samples: {node.samples} | Dist: {node.class_dist}", depth)
        
        for s in imp_steps:
            self._log(s, depth)
        
        # Condições de parada
        if (depth >= self.max_depth or 
            len(np.unique(y)) == 1 or 
            node.samples < self.min_samples_split):
            node.value = Counter(y).most_common(1)[0][0]
            if self.verbose_simple:
                self._log(f"✅ FOLHA: prevê classe {node.value}", depth,
                         "Chegamos a uma decisão final! Todas as amostras aqui serão classificadas como esta classe.")
            else:
                self._log(f"→ LEAF: predict class {node.value}", depth)
            return node
        
        # Encontrar melhor split
        feat, thresh, gain, steps = self._best_split(X, y)
        
        if feat is None or gain <= 1e-7:
            node.value = Counter(y).most_common(1)[0][0]
            if self.verbose_simple:
                self._log(f"⚠ Nenhum split melhorou o grupo → FOLHA: classe {node.value}", depth,
                         "Nenhuma divisão separou melhor as classes, então paramos aqui.")
            else:
                self._log(f"⚠ No valid split → LEAF: class {node.value}", depth)
            return node
        
        # Log do split
        if self.verbose_simple:
            self._log(f"🔍 Melhor divisão encontrada:", depth,
                     "A árvore testou várias formas de dividir os dados e escolheu esta!")
            self._log(f"📏 Dividir por: Feature [{feat}] ≤ {thresh:.3f}", depth,
                     f"Feature {feat} é uma coordenada; threshold {thresh:.3f} é o 'ponto de corte'.")
            self._log(f"💰 Ganho de informação: {gain:.3f}", depth,
                     "Quanto maior o ganho, mais o split ajuda a separar as classes!")
        else:
            self._log(f"🔍 Best split: feature[{feat}] <= {thresh:.3f} | Gain={gain:.4f}", depth)
        
        for s in steps:
            self._log(s, depth)
        
        node.feature_idx = feat
        node.threshold = thresh
        node.gain = gain
        
        # Dividir e recursão
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        
        if self.verbose_simple:
            self._log(f"➡️ Ramo ESQUERDO (amostras com Feature[{feat}] ≤ {thresh:.3f}):", depth,
                     "Agora a árvore repete o processo apenas com as amostras que foram para este lado.")
        else:
            self._log(f"→ Left branch (X[:,{feat}] <= {thresh:.3f}):", depth)
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        
        if self.verbose_simple:
            self._log(f"➡️ Ramo DIREITO (amostras com Feature[{feat}] > {thresh:.3f}):", depth)
        else:
            self._log(f"→ Right branch:", depth)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Treina a árvore e retorna logs"""
        self.logs = []
        if self.verbose_simple:
            self._log("🚀 Iniciando o treinamento da Decision Tree...")
            self._log("💡 A árvore vai aprender a fazer perguntas sobre os dados para separar as classes.", explain=True)
            self._log(f"⚙️ Configurações: critério='{self.criterion}', profundidade máxima={self.max_depth}")
        else:
            self._log("🚀 Starting Decision Tree training...")
            self._log(f"⚙️ Config: criterion='{self.criterion}', max_depth={self.max_depth}")
        
        self._node_id = 0
        self.root = self._build(X, y)
        
        if self.verbose_simple:
            self._log("\n✅ Treinamento concluído!", 'success',
                     "A árvore agora está pronta para classificar novos dados!")
        else:
            self._log("\n✅ Training complete!", 'success')
        
        return self.logs
    
    def predict(self, X):
        """Prediz classes para múltiplas amostras"""
        return np.array([self._predict_one(x, self.root) for x in X])
    
    def _predict_one(self, x, node):
        """Prediz uma única amostra"""
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)
    
    def get_metrics(self, X, y):
        """Calcula métricas do modelo"""
        if self.root is None:
            return {'accuracy': 0, 'max_depth': 0, 'total_nodes': 0, 'leaf_count': 0}
        
        preds = self.predict(X)
        acc = np.mean(preds == y)
        
        def count(n):
            if n is None: return 0, 0
            if n.is_leaf(): return 1, 1
            tl, ll = count(n.left)
            tr, lr = count(n.right)
            return 1 + tl + tr, ll + lr
        
        total, leaves = count(self.root)
        max_d = max(n.depth for n in self._all_nodes())
        
        return {
            'accuracy': float(acc),
            'max_depth': max_d,
            'total_nodes': total,
            'leaf_count': leaves
        }
    
    def _all_nodes(self, node=None):
        """Itera sobre todos os nós"""
        if node is None: node = self.root
        if node is None: return []
        result = [node]
        if not node.is_leaf():
            result.extend(self._all_nodes(node.left))
            result.extend(self._all_nodes(node.right))
        return result


# ===== FUNÇÕES DE VISUALIZAÇÃO =====

def plot_boundary(tree, X, y, resolution=150):
    """Plota fronteira de decisão com splits visíveis"""
    import matplotlib.pyplot as plt
    import io, base64
    
    if X.shape[1] != 2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Apenas datasets 2D\nsão suportados", ha='center', va='center')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap='coolwarm', levels=15)
    
    classes = np.unique(y)
    colors = ['#2563eb', '#dc2626', '#16a34a']
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], 
                  c=[colors[i % len(colors)]], 
                  label=f'Classe {int(cls)}',
                  edgecolors='white', s=35, alpha=0.9)
    
    # Linhas de split
    _draw_splits(ax, tree.root, x_min, x_max, y_min, y_max)
    
    ax.set_xlabel('Feature 1 (X₁)', fontsize=9)
    ax.set_ylabel('Feature 2 (X₂)', fontsize=9)
    ax.legend(loc='lower right', fontsize=7)
    ax.set_title('Fronteira de Decisão Aprendida', fontsize=10)
    ax.grid(alpha=0.3)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _draw_splits(ax, node, x_min, x_max, y_min, y_max, depth=0):
    """Desenha splits da árvore"""
    if node is None or node.is_leaf() or depth > 10:
        return
    feat, thresh = node.feature_idx, node.threshold
    if feat == 0:
        ax.axvline(x=thresh, color='gray', linestyle='--', alpha=0.6, linewidth=0.8)
        _draw_splits(ax, node.left, x_min, thresh, y_min, y_max, depth+1)
        _draw_splits(ax, node.right, thresh, x_max, y_min, y_max, depth+1)
    else:
        ax.axhline(y=thresh, color='gray', linestyle='--', alpha=0.6, linewidth=0.8)
        _draw_splits(ax, node.left, x_min, x_max, y_min, thresh, depth+1)
        _draw_splits(ax, node.right, x_min, x_max, thresh, y_max, depth+1)


def tree_to_text(tree, max_depth=5, simple=True):
    """Gera representação textual da árvore"""
    if tree.root is None:
        return "Árvore não treinada."
    
    lines = ["🌳 Estrutura da Árvore de Decisão:\n"]
    
    def traverse(node, prefix="", is_last=True):
        if node is None or node.depth > max_depth:
            return
        conn = "└── " if is_last else "├── "
        
        if node.is_leaf():
            cls = node.class_dist
            if simple:
                lines.append(f"{prefix}{conn}🍃 DECISÃO FINAL: Classe {node.value}")
                lines.append(f"{prefix}{'    ' if is_last else '│   '}💡 {node.samples} amostras chegam aqui → todas classificadas como {node.value}")
            else:
                lines.append(f"{prefix}{conn}🍃 Leaf: Class {node.value} (n={node.samples}, dist={cls})")
        else:
            gain_str = f" | 💰 Gain={node.gain:.3f}" if node.gain is not None else ""
            if simple:
                lines.append(f"{prefix}{conn}🔀 PERGUNTA: Feature [{node.feature_idx}] ≤ {node.threshold:.3f}?{gain_str}")
                lines.append(f"{prefix}{'    ' if is_last else '│   '}📊 {node.samples} amostras aqui | Impureza: {node.impurity:.3f}")
                if node.gain and simple:
                    lines.append(f"{prefix}{'    ' if is_last else '│   '}💡 Esta pergunta separou bem as classes (ganho alto)!")
            else:
                lines.append(f"{prefix}{conn}🔀 Split: X[{node.feature_idx}] ≤ {node.threshold:.3f}{gain_str}")
                lines.append(f"{prefix}{'    ' if is_last else '│   '}📊 n={node.samples}, imp={node.impurity:.4f}")
            
            new_pref = prefix + ("    " if is_last else "│   ")
            if node.left:
                traverse(node.left, new_pref, node.right is None)
            if node.right:
                traverse(node.right, new_pref, True)
    
    traverse(tree.root)
    
    if simple:
        lines.append("\n🔑 Legenda:")
        lines.append("   🔀 = A árvore faz uma pergunta sobre os dados")
        lines.append("   🍃 = Decisão final: todas as amostras aqui recebem esta classe")
        lines.append("   💰 = Quanto maior, melhor a pergunta separou as classes")
    
    return "\n".join(lines)


# ===== DATASETS DE DEMONSTRAÇÃO =====

def generate_dataset(type_='moon', n=100, noise=0.15):
    """Gera datasets sintéticos 2D com contexto"""
    np.random.seed(42)
    
    if type_ == 'moon':
        n_half = n // 2
        angle = np.linspace(0, np.pi, n_half)
        x1 = np.column_stack([np.cos(angle) + np.random.normal(0, noise, n_half),
                             np.sin(angle) + np.random.normal(0, noise, n_half)])
        x2 = np.column_stack([np.cos(angle) + 1 + np.random.normal(0, noise, n_half),
                             -np.sin(angle) + 0.5 + np.random.normal(0, noise, n_half)])
        X = np.vstack([x1, x2])
        y = np.array([0]*n_half + [1]*n_half)
        
    elif type_ == 'circle':
        n_half = n // 2
        angle = np.random.uniform(0, 2*np.pi, n_half)
        x1 = np.column_stack([np.cos(angle) + np.random.normal(0, noise, n_half),
                             np.sin(angle) + np.random.normal(0, noise, n_half)])
        x2 = np.column_stack([2*np.cos(angle) + np.random.normal(0, noise, n_half),
                             2*np.sin(angle) + np.random.normal(0, noise, n_half)])
        X = np.vstack([x1, x2])
        y = np.array([0]*n_half + [1]*n_half)
        
    else:  # linear
        X = np.random.randn(n, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    return X.astype(float), y.astype(int)

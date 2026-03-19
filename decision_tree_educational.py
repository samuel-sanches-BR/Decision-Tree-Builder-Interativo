"""
Decision Tree Educacional - Implementação didática para navegador (Pyodide)
- Cálculos explícitos de Gini/Entropia com logs
- Visualizações com matplotlib compatíveis com Pyodide
- Interface simples para ensino de Machine Learning
"""

import numpy as np
from collections import Counter
import math


class DecisionTreeNode:
    """Nó da árvore com metadados para visualização educacional"""
    def __init__(self, depth=0, node_id=0):
        self.depth = depth
        self.node_id = node_id
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None  # Classe prevista (se folha)
        self.samples = 0
        self.impurity = None
        self.class_dist = {}
        self.gain = None
        
    def is_leaf(self):
        return self.value is not None


class EducationalDecisionTree:
    """Árvore de decisão com transparência nos cálculos"""
    
    def __init__(self, criterion='gini', max_depth=3, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self._node_id = 0
        self.logs = []
    
    def _log(self, msg, indent=0):
        """Adiciona mensagem ao log com indentação"""
        self.logs.append("  " * indent + msg)
    
    def _gini(self, y):
        """Calcula Gini Impurity com passos explicativos"""
        if len(y) == 0:
            return 0.0, []
        counts = Counter(y)
        n = len(y)
        gini = 1.0
        steps = [f"• Contagem: {dict(counts)}"]
        for label, cnt in counts.items():
            p = cnt / n
            gini -= p ** 2
            steps.append(f"  Classe {label}: p={p:.3f} → p²={p**2:.4f}")
        steps.append(f"→ Gini = {gini:.4f}")
        return gini, steps
    
    def _entropy(self, y):
        """Calcula Entropia com passos explicativos"""
        if len(y) == 0:
            return 0.0, []
        counts = Counter(y)
        n = len(y)
        ent = 0.0
        steps = [f"• Contagem: {dict(counts)}"]
        for label, cnt in counts.items():
            p = cnt / n
            if p > 0:
                term = -p * math.log2(p)
                ent += term
                steps.append(f"  Classe {label}: p={p:.3f} → -p·log₂(p)={term:.4f}")
        steps.append(f"→ Entropia = {ent:.4f}")
        return ent, steps
    
    def _impurity(self, y):
        """Função unificada para calcular impureza"""
        if self.criterion == 'gini':
            return self._gini(y)
        return self._entropy(y)
    
    def _information_gain(self, y, y_left, y_right):
        """Calcula ganho de informação para um split candidato"""
        parent_imp, parent_steps = self._impurity(y)
        n = len(y)
        n_l, n_r = len(y_left), len(y_right)
        
        if n_l == 0 or n_r == 0:
            return -1, parent_steps + ["⚠ Split inválido: grupo vazio"]
        
        imp_l, _ = self._impurity(y_left)
        imp_r, _ = self._impurity(y_right)
        weighted = (n_l/n) * imp_l + (n_r/n) * imp_r
        gain = parent_imp - weighted
        
        steps = parent_steps + ["\nApós split:"]
        steps.append(f"  Esquerda (n={n_l}): impureza={imp_l:.4f}")
        steps.append(f"  Direita (n={n_r}): impureza={imp_r:.4f}")
        steps.append(f"  Ponderada: ({n_l}/{n})×{imp_l:.4f} + ({n_r}/{n})×{imp_r:.4f} = {weighted:.4f}")
        steps.append(f"Gain = {parent_imp:.4f} - {weighted:.4f} = {gain:.4f}")
        return gain, steps
    
    def _best_split(self, X, y):
        """Encontra o melhor split testando todas as possibilidades"""
        best_gain, best_feat, best_thresh = -1, None, None
        best_steps = []
        
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left = X[:, feat] <= thresh
                right = ~left
                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue
                gain, steps = self._information_gain(y, y[left], y[right])
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
                    best_steps = steps
        
        return best_feat, best_thresh, best_gain, best_steps
    
    def _build(self, X, y, depth=0):
        """Constrói a árvore recursivamente"""
        node = DecisionTreeNode(depth=depth, node_id=self._node_id)
        self._node_id += 1
        
        node.samples = len(y)
        node.class_dist = dict(Counter(y))
        node.impurity, imp_steps = self._impurity(y)
        
        # Log do nó atual
        self._log(f"\nNó #{node.node_id} (profundidade {depth})", depth)
        self._log(f"Amostras: {node.samples} | Distribuição: {node.class_dist}", depth)
        for s in imp_steps:
            self._log(s, depth)
        
        # Condições de parada → folha
        if (depth >= self.max_depth or 
            len(np.unique(y)) == 1 or 
            node.samples < self.min_samples_split):
            node.value = Counter(y).most_common(1)[0][0]
            self._log(f"FOLHA: prevê classe {node.value}", depth)
            return node
        
        # Encontrar melhor split
        feat, thresh, gain, steps = self._best_split(X, y)
        
        if feat is None or gain <= 1e-7:
            node.value = Counter(y).most_common(1)[0][0]
            self._log(f"⚠ Sem split válido → FOLHA: classe {node.value}", depth)
            return node
        
        # Log do split
        self._log(f"Melhor split: feature[{feat}] <= {thresh:.3f} | Gain={gain:.4f}", depth)
        for s in steps:
            self._log(s, depth)
        
        node.feature_idx = feat
        node.threshold = thresh
        node.gain = gain
        
        # Dividir e recursão
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        
        self._log(f"Ramo ESQUERDO (X[:,{feat}] <= {thresh:.3f}):", depth)
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        
        self._log(f"Ramo DIREITO (X[:,{feat}] > {thresh:.3f}):", depth)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Treina a árvore e retorna logs"""
        self.logs = ["Iniciando treinamento da Decision Tree..."]
        self.logs.append(f"Config: criterion='{self.criterion}', max_depth={self.max_depth}, min_samples={self.min_samples_split}")
        self._node_id = 0
        self.root = self._build(X, y)
        self.logs.append("\nTreinamento concluído!")
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
        """Calcula métricas do modelo treinado"""
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
        """Itera sobre todos os nós da árvore"""
        if node is None: node = self.root
        if node is None: return []
        result = [node]
        if not node.is_leaf():
            result.extend(self._all_nodes(node.left))
            result.extend(self._all_nodes(node.right))
        return result


# ===== FUNÇÕES DE VISUALIZAÇÃO =====

def plot_boundary(tree, X, y, resolution=150):
    """Plota fronteira de decisão e dados (retorna base64 para embed)"""
    import matplotlib.pyplot as plt
    import io, base64
    
    if X.shape[1] != 2:
        # Plot de erro para dimensões inválidas
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Apenas datasets 2D\nsão suportados para\nvisualização", 
               ha='center', va='center', fontsize=10)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    
    # Grid para fronteira
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Predições no grid
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap='coolwarm', levels=15)
    
    # Dados
    classes = np.unique(y)
    colors = ['#2563eb', '#dc2626', '#16a34a', '#9333ea']
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], 
                  c=[colors[i % len(colors)]], 
                  label=f'Classe {int(cls)}',
                  edgecolors='white', s=35, alpha=0.9)
    
    # Linhas de split
    _draw_splits(ax, tree.root, x_min, x_max, y_min, y_max)
    
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.legend(loc='lower right', fontsize=7)
    ax.set_title('Fronteira de Decisão', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Converter para base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _draw_splits(ax, node, x_min, x_max, y_min, y_max, depth=0):
    """Desenha recursivamente as linhas de split da árvore"""
    if node is None or node.is_leaf() or depth > 10:
        return
    
    feat, thresh = node.feature_idx, node.threshold
    
    if feat == 0:  # Split vertical
        ax.axvline(x=thresh, color='gray', linestyle='--', alpha=0.6, linewidth=0.8)
        _draw_splits(ax, node.left, x_min, thresh, y_min, y_max, depth+1)
        _draw_splits(ax, node.right, thresh, x_max, y_min, y_max, depth+1)
    else:  # Split horizontal
        ax.axhline(y=thresh, color='gray', linestyle='--', alpha=0.6, linewidth=0.8)
        _draw_splits(ax, node.left, x_min, x_max, y_min, thresh, depth+1)
        _draw_splits(ax, node.right, x_min, x_max, thresh, y_max, depth+1)


def tree_to_text(tree, max_depth=5):
    """Gera representação textual da árvore para exibição"""
    if tree.root is None:
        return "Árvore não treinada."
    
    lines = ["🌳 Estrutura da Árvore:\n"]
    
    def traverse(node, prefix="", is_last=True):
        if node is None or node.depth > max_depth:
            return
        conn = "└── " if is_last else "├── "
        
        if node.is_leaf():
            cls = node.class_dist
            lines.append(f"{prefix}{conn}🍃 Folha: Classe {node.value} (n={node.samples}, dist={cls})")
        else:
            gain_str = f" | Gain={node.gain:.4f}" if node.gain is not None else ""
            lines.append(f"{prefix}{conn}Split: X[{node.feature_idx}] ≤ {node.threshold:.3f}{gain_str}")
            lines.append(f"{prefix}{'    ' if is_last else '│   '}n={node.samples}, impureza={node.impurity:.4f}")
            
            new_pref = prefix + ("    " if is_last else "│   ")
            if node.left:
                traverse(node.left, new_pref, node.right is None)
            if node.right:
                traverse(node.right, new_pref, True)
    
    traverse(tree.root)
    return "\n".join(lines)


# ===== DATASETS DE DEMONSTRAÇÃO =====

def generate_dataset(type_='moon', n=100, noise=0.15):
    """Gera datasets sintéticos 2D para demonstração"""
    np.random.seed(42)
    
    if type_ == 'moon':
        # Two moons
        n_half = n // 2
        angle = np.linspace(0, np.pi, n_half)
        x1 = np.column_stack([np.cos(angle) + np.random.normal(0, noise, n_half),
                             np.sin(angle) + np.random.normal(0, noise, n_half)])
        x2 = np.column_stack([np.cos(angle) + 1 + np.random.normal(0, noise, n_half),
                             -np.sin(angle) + 0.5 + np.random.normal(0, noise, n_half)])
        X = np.vstack([x1, x2])
        y = np.array([0]*n_half + [1]*n_half)
        
    elif type_ == 'circle':
        # Concentric circles
        n_half = n // 2
        angle = np.random.uniform(0, 2*np.pi, n_half)
        x1 = np.column_stack([np.cos(angle) + np.random.normal(0, noise, n_half),
                             np.sin(angle) + np.random.normal(0, noise, n_half)])
        x2 = np.column_stack([2*np.cos(angle) + np.random.normal(0, noise, n_half),
                             2*np.sin(angle) + np.random.normal(0, noise, n_half)])
        X = np.vstack([x1, x2])
        y = np.array([0]*n_half + [1]*n_half)
        
    else:  # linear
        # Linearmente separável
        X = np.random.randn(n, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    return X.astype(float), y.astype(int)

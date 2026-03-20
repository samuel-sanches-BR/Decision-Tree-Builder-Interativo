# 🌳 Decision Tree Builder — Aprendizado Interativo (Web Demo)

Este repositório contém o código-fonte de uma aplicação web interativa que demonstra o funcionamento de uma **Árvore de Decisão (Decision Tree)**
com ênfase na matemática por trás do processo de aprendizado: **cálculo de impureza**, **ganho de informação** e **escolha do melhor split**.

O grande diferencial deste projeto é que todo o processamento matemático, o treinamento da árvore e a geração dos gráficos são executados
**nativamente no navegador do usuário**, sem a necessidade de um backend em Python ou de qualquer instalação local.

## Aplicação ao Vivo

Você pode testar a aplicação diretamente pelo link abaixo:

**[Acessar o Projeto (GitHub Pages)](https://samuel-sanches-br.github.io/Decision-Tree-Builder-Interativo/)**

---

## Sobre o Projeto

A aplicação permite configurar e treinar uma árvore de decisão do zero, acompanhando cada decisão de split em tempo real.
O usuário escolhe o critério de impureza, a profundidade máxima e o dataset — e a árvore exibe todos os cálculos intermediários,
tornando o processo de aprendizado completamente transparente e auditável.

### Funcionalidades Principais

* **Execução Client-Side:** todo o código Python roda 100% no navegador via WebAssembly, sem servidor.
* **2 Critérios de Impureza:** Gini e Entropia — com fórmula renderizada em LaTeX e explicação adaptável ao nível do usuário.
* **Cálculos passo a passo:** cada nó exibe o número de amostras, a distribuição das classes, o cálculo do critério e o ganho de informação do melhor split.
* **3 Datasets embutidos:** Two Moons (não-linear), Circles (radial) e Linear (separável por reta) — gerados diretamente no navegador.
* **Visualização dos Dados:** gráfico de dispersão das classes antes do treinamento.
* **Fronteira de Decisão:** mapa de cores mostrando as regiões classificadas pela árvore após o treino, com as linhas de corte sobrepostas.
* **Estrutura da Árvore:** representação textual hierárquica com nós de decisão e folhas, legível em modo simples ou técnico.
* **Métricas em tempo real:** acurácia, profundidade efetiva, número total de nós e número de folhas.
* **Dois modos de explicação:** 🎓 Simples (linguagem acessível, ideal para iniciantes) e ⚙️ Técnico (terminologia precisa de ML).

---

## Como Usar

| Passo | Ação |
| ----- | ---- |
| 1 | Escolha o **critério** de impureza: Gini ou Entropia |
| 2 | Ajuste a **profundidade máxima** com o controle deslizante (1–6) |
| 3 | Selecione o **dataset**: 🌙 Moons, ⭕ Circles ou 📈 Linear |
| 4 | Escolha o **modo de explicação**: 🎓 Simples ou ⚙️ Técnico |
| 5 | Clique em **▶ Treinar** e acompanhe os cálculos passo a passo |
| 6 | Analise a **fronteira de decisão**, a **estrutura da árvore** e as **métricas** |

---

## Conceitos Demonstrados

| Conceito | O que o projeto mostra |
| -------- | ---------------------- |
| **Impureza de Gini** | `Gini(D) = 1 − Σpᵢ²` — calculado para cada nó com os valores reais |
| **Entropia** | `H(D) = −Σpᵢ · log₂(pᵢ)` — com o número de bits de incerteza |
| **Ganho de Informação** | `Gain = Impureza(pai) − média_ponderada(Impureza filhos)` |
| **Escolha do split** | Busca exaustiva sobre todas as features e todos os limiares possíveis |
| **Critérios de parada** | Profundidade máxima, nó puro, mínimo de amostras |
| **Fronteira de decisão** | Regiões axis-aligned que ilustram a natureza retangular das árvores |
| **Overfitting vs. Generalização** | Efeito visível ao aumentar a profundidade nos datasets não-lineares |

---

## Tecnologias Utilizadas

* **Front-end:** HTML5, CSS3, JavaScript
* **Execução Python no Web:** [Pyodide](https://pyodide.org/) (CPython via WebAssembly)
* **Renderização de Equações:** [KaTeX](https://katex.org/)
* **Matemática e Álgebra Linear:** `numpy`
* **Visualização de Dados:** `matplotlib`

---

## Estrutura do Repositório

```
├── index.html                      # Interface web e código Python embutido
├── decision_tree_educational.py    # Módulo Python (versão standalone para referência)
├── README.md
└── LICENSE
```

`index.html` carrega o Pyodide dinamicamente e executa o código Python embutido via WebAssembly.
O módulo `decision_tree_educational.py` contém a implementação da árvore (`EducationalDecisionTree`),
as funções de geração de datasets (`generate_dataset`) e de visualização (`plot_boundary`, `tree_to_text`).
O resultado do treinamento é um JSON com os logs de cálculo e as imagens em base64, renderizados progressivamente pelo JavaScript.

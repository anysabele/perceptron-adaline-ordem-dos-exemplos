# Perceptron e Adaline: o efeito da ordem dos exemplos

Implementação **do zero, só com NumPy**, do Perceptron e do Adaline para investigar uma pergunta simples:

> **Os algoritmos chegam sempre à mesma solução?**

A variável experimental é a **ordem de apresentação dos exemplos** durante o treinamento: ordem fixa × reembaralhada a cada época. 

## O que foi feito

- **Dados sintéticos linearmente separáveis**, com 2 atributos e 2 classes, gerados por distribuições uniformes em regiões disjuntas (sem sobreposição, por construção):
  - Classe 0: `uniform([-3.0, -3.0], [-0.5, -0.5])`
  - Classe 1: `uniform([0.5, 0.5], [3.0, 3.0])`
  - 200 pontos por classe (400 no total), com divisão 70% treino / 30% teste.
- **Padronização** dos atributos com média e desvio padrão calculados só no treino.
- **Perceptron** (Rosenblatt): atualização online, exemplo a exemplo, com saída limiarizada (`z ≥ 0`) e taxa de aprendizagem 0,1.
- **Adaline** (regra Delta / Widrow-Hoff): atualização online com a **saída linear** (sem limiar) e taxa de aprendizagem 0,01. O limiar (`z ≥ 0,5`) só é aplicado na predição.
- **Critério de parada**: orçamento fixo de **100 épocas** para todas as execuções.
- **Repetições**: 5 sementes (10, 20, 30, 40, 50) controlando o embaralhamento.

### Equações

| | Perceptron | Adaline |
|---|---|---|
| Saída no treino | `ŷ = 1 se w·x + b ≥ 0, senão 0` | `z = w·x + b` (linear) |
| Erro | `e = y − ŷ` | `e = y − z` |
| Atualização | `w ← w + η·e·x` (só se errar) | `w ← w + η·(y − z)·x` (a cada exemplo) |

## Resultados

Média ± desvio padrão nas 5 sementes. A acurácia foi **100% no treino e no teste em todos os cenários**.

### Perceptron

| | w1 | w2 | bias |
|---|---|---|---|
| Ordem fixa | 0,1687 ± 0,0000 | 0,1801 ± 0,0000 | 0,0000 ± 0,0000 |
| Ordem embaralhada | 0,1267 ± 0,0463 | 0,1109 ± 0,0162 | 0,0000 ± 0,0000 |

### Adaline

| | w1 | w2 | bias | MSE final |
|---|---|---|---|---|
| Ordem fixa | 0,2335 ± 0,0000 | 0,2692 ± 0,0000 | 0,4893 ± 0,0000 | 0,022979 ± 0,000000 |
| Ordem embaralhada | 0,2314 ± 0,0033 | 0,2627 ± 0,0037 | 0,4898 ± 0,0070 | 0,022999 ± 0,000091 |

> Os valores de "épocas = 100" refletem o orçamento fixo do protocolo, não o momento em que o algoritmo convergiu.

## Conclusões

- **Os algoritmos não chegam necessariamente à mesma solução.** Mesmo com acurácia idêntica, ordens diferentes levam a pesos e fronteiras de decisão diferentes.
- **Ordem fixa** gera resultados idênticos entre sementes (a semente só importa quando há embaralhamento).
- **Perceptron**: o mais sensível à ordem. Como só atualiza quando erra, mudar a sequência muda o caminho percorrido e os dados separáveis admitem vários hiperplanos válidos.
- **Adaline**: bem mais estável. Ele atualiza com o erro contínuo e minimiza o erro quadrático médio, então as variações nos pesos, no bias e no MSE são mínimas.

## Conteúdo do notebook

1. Geração dos dados
2. Divisão treino/teste e padronização
3. Implementação do Perceptron
4. Implementação do Adaline
5. Desenho experimental do efeito da ordem
6. Resultados por semente
7. Média e desvio padrão
8. Evolução do erro (nº de erros no Perceptron, MSE no Adaline)
9. Fronteiras de decisão das 5 sementes
10. Matrizes de confusão
11. Boxplots (pesos, bias e épocas)
12. Discussão e conclusão

## Como executar

```bash
git clone https://github.com/<seu-usuario>/<nome-do-repositorio>.git
cd <nome-do-repositorio>
pip install numpy matplotlib jupyter
jupyter notebook Projeto_1_Perceptron_e_Adaline.ipynb
```

Não há dados externos: o conjunto é gerado no próprio notebook (semente 42), então os resultados são reprodutíveis.

## Estrutura

```
.
├── Projeto_1_Perceptron_e_Adaline.ipynb
└── README.md
```

## Tecnologias

Python · NumPy · Matplotlib

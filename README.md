# TREINANDO UMA REDE CONVOLUCIONAL COM O DATASET MNIST

Em um processo seletivo para uma vaga de estágio, me foi apresentado o seguinte problema:
```
Vários desafios de visão computacional consistem em classificar imagens, logo saber criar classificadores é um skill muito importante para quem trabalha com visão computacional. 

Nesse notebook da FastAI (https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb) são ensinados alguns conceitos e estratégias importantes para criar um classificador capaz de diferenciar os dígitos 3 e 7 do dataset MNIST.

Para este desafio técnico, queremos que você crie um classificador capaz de diferenciar os números 0 e 5 do dataset MNIST.
```

Para resolver o problema utilizei a biblioteca PyTorch, que possui muitas abstrações úteis para a resolução de problemas do tipo. Fiz um script simples mas que resolve bem o problema, detalharei minha linha de raciocínio por meio deste documento.

## A escolha do tipo de rede

Optei por uma rede neural convolucional, já que ela é extremamente útil para problemas de classificação e até de detecção de objetos.

Em meu script, por meio da biblioteca PyTorch, baixei o dataset MNIST (é importante notar que um problema interno da versão de PyTorch que uso gera um aviso ao realizar tal tarefa) e configurei uma rede convolucional de acordo com o problema. Mantive apenas dois parâmetros livres: learning rate e número de epochs.

## O ambiente de execução

Visando permitir uma reprodutibilidade dos resultados que obtive, defini todas as seeds do script como 0 e desativei todo o tipo de operação não determinística. É importante notar que todo o script foi executado em uma máquina com placa gráfica GTX 1050Ti, 16GB de RAM e um processador Intel Core i7 de sétima geração com o sistema operacional Windows 10. As versões dos módulos utilizados podem ser verificados no arquivo `requirements.txt` disponibilizado.

## O desafio

O desafio encontrado foi superar a acurácia de 99,70% alcançada pelo algoritmo do notebook do fast.ai. Para tanto, utilizei um otimizador ADAM e, para fazer uma comparação justa, utilizei a mesma função de loss que o notebook apresentado, Cross Entropy Loss.

## Os testes

Para realizar os testes defini algumas regras:
* Diminuiria o learning rate sempre que o loss começasse a oscilar muito;
* Caso o loss não oscilasse muito, mas a acurácia fosse baixa, aumentaria o número de epochs;
* Caso a variação do loss fosse muito baixa, aumentaria o learning rate;
* Caso a variação do loss em adição de épocas fosse muito baixa, manteria a menor quantidade de épocas;

Seguindo tal "algoritmo" cheguei na seguinte tabela:

| Learning Rate | Epochs | Accuracy |
|---------------|--------|----------|
| 1e-6          | 1      | 47.65%   |
| 1e-5          | 1      | 95.62%   |
| 1e-5          | 2      | 97.97%   |
| 1e-4          | 2      | 99.20%   |
| 1e-3          | 2      | 99.73%   |
| 1e-2          | 2      | 52.35%   |
| 1e-3          | 4      | 99.84%   |
| 1e-3          | 8      | 99.95%   |
| 1e-3          | 10     | 99.63%   |

Assumi que o valor ideal para learning rate fosse $10^{-3}$ e o número ideal de épocas fosse 8. A implementação pode ser considerada um sucesso, pois conseguimos um número muito próximo de 100% para a acurácia. A comparação com o método do fast.ai talvez não seja justa, uma vez que, mesmo utilizando a mesma função de loss, estamos trabalhando com um conjunto diferente de imagens (ainda que originárias do mesmo data set).

O código fonte em `net.py` possui anotações dos detalhes de implementação do algoritmo. E as imagens em `output/graph/` podem ser utilizadas como referência para os treinos. As imagens seguem o padrão `mnist_N_M.png` onde N é o número de epochs e M é $-log_{10}(lr)$.

Acredito que o desafio tenha sido interessante e a experiência proveitosa. Estou a disposição para quaisquer questionamentos acerca dos resultados obtidos ou da implementação.

Até breve! :grinning:
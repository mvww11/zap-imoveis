# Regressão: Estimando o valor de venda de um imóvel
Esse é um projeto completo de data science. Inclui etapas como data Scrapping com BeautifulSoup no [zapimoveis.com.br](https://www.zapimoveis.com.br/), tratamento de missing data, análise exploratória de dados, modelagem, otimização dos hiperparâmetros e explicação das decisões do modelo.

Nessa página você encontra um resumo do projeto. A versão completa está separada nos arquivos [zap scrapping.ipynb](zap%20scrapping.ipynb), [data wrangling.ipynb](data%20wrangling.ipynb), [EDA.ipynb](EDA.ipynb) e [zap-modelling.ipynb](zap-modelling.ipynb).

Nosso cliente possui um imóvel situado na Avenida Oswaldo Cruz, Flamengo, RJ. Ele quer saber qual é o valor de venda desse imóvel. Para isso, coletamos dados no [zapimoveis.com.br](https://www.zapimoveis.com.br/) sobre outros apartamentos à venda na mesma região, e treinamos um modelo de Gradient Boosting para prever o valor do imóvel, com base em informações como área do imóvel, logradouro, número de quartos e vagas de garagem.

Nosso modelo ótimo alcançou um erro absoluto médio aproximado de R$150 mil. Para dar dimensão, a média do valor dos imóveis em nosso data set é de R$1,5 milhão.

A previsão que o modelo fez para o imóvel de nosso cliente foi de R$1,18 milhão.

## Resumo do Projeto
* Objetivo: criar um modelo de previsão da probabilidade de uma reserva de hotel ser cancelada.
* Nosso modelo xgboost final alcançou um recall de 92% em data points nunca vistos por ele.
* Dados: 80 mil reservas de um hotel situado em Lisboa, Portugal.
* Análise exploratória de dados mostrou que a renda é o fator mais relevante para a previsão da nota.
* Feature engineering: criei duas features novas: uma que indica a renda per capita (por residente no domicílio) do candidato e outra que indica a escolaridade máxima entre pai e mãe.
* Benchmark model com XGBoost e LightGBM para análise de importâncias relativas entre features e feature selection.
* Refinamento do modelo: procura por hiperparâmetros ótimos usando bayesian search.
* Interpretação do modelo: expliquei quais são as decisões que o modelo faz para chegar a uma previsão. Para isso, usei valores SHAP.
* Deploy serverless do modelo no [AWS Lambda](https://aws.amazon.com/lambda/) e criação de um [bot do Telegram](https://telegram.org/blog/bot-revolution) que permite que qualquer pessoa faça a previsão da sua nota no ENEM usando nosso modelo.

## Recursos utilizados
**Python**: Versão 3.7<br>
**Pacotes Python**: beautifulsoup, numpy, pandas, matplotlib, seaborn, xgboost, hyperopt, joblib, shap<br>
**Bayesian optimization**: [[1]](https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a) [[2]](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex.html)<br>
**Explicando o modelo com SHAP**: [[1]](https://medium.com/@gabrieltseng/interpreting-complex-models-with-shap-values-1c187db6ec83) [[2]](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30) [[3]](https://towardsdatascience.com/black-box-models-are-actually-more-explainable-than-a-logistic-regression-f263c22795d) [[4]](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d)

## Obtenção dos dados
Para obter os dados, fizemos data scrapping no site de classificados de imóveis [zapimoveis.com.br](https://www.zapimoveis.com.br/). Utilizamos a biblioteca beautifulsoup, que extrai as informações a partir do HTML da página. O procedimento completo pode ser visto no arquivo [zap scrapping.ipynb](zap%20scrapping.ipynb).

Escolhemos coletar apenas imóveis situados na própria Avenida Oswaldo Cruz ou em ruas próximas, pois seriam estatísticamente próximas ao tipo de imóvel que queremos modelar.

As features que conseguimos extrair são:
|Variável                       |Tipo     |Descrição |
|:------------------------------|:---------|:-----------|
|Price                          |float     | Valor de venda anunciado do imóvel, em R$ |
|Address                      |string    | Logradouro onde o imóvel está situado |
|Area                         |float    | Área do imóvel, em m²|
|Baths                        |int | Quantidade de banheiros no imóvel |
|Condominio             |float | Valor da cota de condomínio do imóvel, em R$ |
|Dorms                       |int    | Quantidade de dormitórios no imóvel|
|IPTU                            |float    | Valor da cota de IPTU do imóvel, em R$ |
|Parking Slots              |int    | Quantidade de vagas de garagem na escritura do imóvel |

Ao final do data scrappping, ficamos com 2889 imóveis em nosso data set.

## Data Cleaning and Wrangling
Após extrair os dados, precisei fazer uma série de transformações para que ficassem apropriados para serem utilizados na análise exploratória e no treinamento dos modelos. Confira a etapa completa em [data wrangling.ipynb](data%20wrangling.ipynb).
* Removi cerca de 790 imóveis que estavam fora da nossa região de interesse. Acabaram aparecendo em nosso data set porque o Zap Imóveis inclui no resultado diversos anúncios patrocinados que não satisfazem os critérios do filtro da busca.
* Diversos valores do data set continham caracteres de formatação indesejados, como os símbolos "\n", "R$" ou "m²". Eliminei esses símbolos, deixando apenas os valores relevantes.
* Todos os nossos valores estavam como string. Converti as features numéricas para datatypes do tipo int ou float.
* Removi 92 linhas duplicadas, que acabaram aparecendo em nosso data set porque anúncios patrocinados são mostrados em mais de uma página do resultado da busca do Zap Imóveis.
* Cerca de 35% dos dados sobre Vagas de Garagem eram nulos. Transformei-os em valores zero. Isso porque, no Zap, apartamentos que não possuem vagas de garagem simplesmente não exibem essa informação no anúncio. Desse modo, nosso scrapper não pôde encontrá-la.
* Removi cerca de 100 apartamentos que não possuíam informações sobre Quantidade de banheiros, quartos, ou valor do condomínio.
* Após o Data Cleaning, ficamos com 1922 apartamentos em nosso data set.

## Análise Exploratória de Dados e Feature Engineering
Após o tratamento de missing data, ficamos com 78879 data points. Entre essas reservas, 41% foram canceladas. Isso indica que o cancelamento de reservas tem um impacto muito grande no business. Se conseguirmos diminuir esse percentual, o potencial de geração de lucro para o negócio é enorme.

Abaixo estão ilustrados alguns insights observados na Análise Exploratória, e as Feature Engineering realizadas. A análise completa está no arquivo [EDA.ipynb](EDA.ipynb).
* A proporção de cancelamentos era maior em reservas feitas por clientes de Portugal.
* A proporção de cancelamentos era menor em reservas feitas por clientes da União Europeia que não de Portugal.
* As duas informações acima me levaram a fazer 2 Feature Engineering: **isPRT**: a reserva foi feita por um cliente de Portugal? **isEU**: a reserva foi feita por um cliente da união Europeia? Confira o gráfico abaixo.
* 40% das reservas possuíam algum tipo de pedido especial, e tinham uma taxa de cancelamento 2.5x menor que reservas sem nenhum pedido especial.
* Reservas que possuíam apenas dias de final de semana tinham uma taxa de cancelamento menor
* A informação acima me levou a criar a seguinte feature: **isOnlyWeekend**: a reserva possui apenas dias de final de semana?
<img src='isPRT_cancel.png' width="400">



## Data Leakage
Algumas features de nosso data set foram eliminadas antes do treinamento do modelo, para evitar data leakage. Por exemplo, a coluna 'ReservationStatus' ( que possui 3 valores possíveis: 'cancelled', 'no-show', 'check-out') determina completamente se a reserva foi cancelada ou não. Entretanto, quando o modelo for colocado em produção, ele tentará prever se a reserva será cancelada ANTES de termos a informação sobre o 'ReservationStatus'. Por isso, nosso modelo não pode usar essa informação no treinamento. O mesmo vale para a coluna 'ReservationStatusDate' e para a coluna 'AssignedRoomType'. Logo, essas 3 colunas foram eliminadas da análise.

## Modelagem e split dos dados
Usaremos um modelo de Gradient Boosting com a implementação da biblioteca xgboost. O processo completo de modelagem pode ser visto no arquivo [modeling.ipynb](modeling.ipynb).

Faremos um split de 60/20/20% dos dados em conjuntos de treinamento, validação e teste, respectivamente. O conjunto de treinamento será aquele em que ajustaremos os parâmetros treináveis de nosso modelo. Como treinaremos vários modelos, que diferem por seus hiperparâmetros, usaremos o conjunto de validação avaliar qual é o melhor entre eles. Já o conjunto de teste será usado uma única vez no final do projeto para estimar a performance que o modelo terá em produção. Desse modo, o conjunto de teste será composto de pontos nunca vistos pelo modelo durante seu treinamento e refinamento.

A métrica que usaremos para avaliar qual é o melhor modelo será a área sob a curva ROC, conhecida como AUC (area under curve). Quanto maior o valor dessa métrica, melhor é o trade-off que teremos entre positivos verdadeiros e positivos falsos (i.e., entre o modelo acertar quais reservas serão canceladas e não errar as reservas não seriam canceladas).

## Benchmark
Um modelo inicial foi treinado com xgboost, usando os hiperparâmetros default. Para não precisar tratar o número de árvores no modelo como um hiperparâmetro a ser otimizado, definimos que interromperíamos o treinamento quando nossa métrica de auc não apresentasse melhora por 30 árvores seguidas (early_stopping_rounds).

Nosso modelo de benchmark alcançou um AUC de 0.946 e um Recall de 82.54%. Ou seja, o modelo previu corretamente 82.54% das reservas que seriam canceladas.

## Refinamento do modelo
O processo de refinamento consiste em treinar diferentes modelos, que diferem pelos seus hiperparâmetros, e utilizar nosso conjunto de validação para verificar qual dos modelos faz a melhor previsão. Compararemos a qualidade dos modelos pela métrica do AUC.

Para buscar os melhores hiperparâmetros, utilizaremos o [Bayesian optimization](https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a). Nesse método, sorteamos aleatoriamente valores para os hiperparâmetros de acordo com uma distribuição de probabilidade. O Bayesian optimization utiliza iterativamente os resultados que vão sendo obtidos para explorar mais intensamente os intervalos de valores mais promissores, para cada hiperparâmetro. Assim, a cada novo modelo treinado, é atualizada a distribuição de probabilidade dos valores associados a cada hiperparâmetro. A implementação do Bayesian optimization que utilizaremos aqui é a do pacote [hyperopt](https://github.com/hyperopt/hyperopt).

Como o pacote [hyperopt](https://github.com/hyperopt/hyperopt) precisa de uma função para MINIMIZAR, vamos definí-la como 1 - AUC. Assim, minimizando o AUC, estamos maximizando o AUC.

Após treinarmos 320 modelos diferentes, o melhor entre eles apresentou um AUC de 0.952 e um Recall de 83.42%.

O processo completo de refinamento também pode ser visto no arquivo [modeling.ipynb](modeling.ipynb).

## Avaliando Overfitting
nosso modelo alcançou um f1_score de 96.62% no conjunto de treinamento, contra 85.26% no conjunto de validação. Isso aponta que nosso modelo tem um certo grau de overfitting, uma vez que sua performance é consideravelmente melhor nos pontos em que foi treinado.

Para minimizar o overfitting podemos otimizar certos hiperparâmetros do XGBoosting com que ainda não trabalhamos, como o subsample. Outra estratégia para combater o problema é aumentar o tamanho de nosso conjunto de treinamento. Para isso, uma possibilidade é fazermos k-fold validation, ao invés de usar um conjunto separado para validação.

## Otimizando para o Recall
Acredito que o recall tenha uma relevância grande para o projeto. Isso porque o recall mede o percentual de acertos do modelo nas reservas que são canceladas, de fato. E se o hotel sabe de antemão que uma reserva tem alta probabilidade de cancelamento, pode agir para evitar esse cancelamento (oferecendo algum benefício ao cliente, por exemplo). Isso tem um grande potencial de impacto no negócio, uma vez que, como vimos, 41% de todas as reservas são canceladas, em média.

Até agora, para que nosso modelo previsse que uma reserva seria cancelada, era necessário que a probabilidade de cancelamento calculada por ele para essa reserva chegasse a 50%. Podemos otimizar o recall diminuindo esse threshold de probabilidade. Assim, por exemplo, uma reserva com 45% de probabilidade de cancelamento já seria prevista como cancelada por nosso modelo.

Temos um trade-off, no entanto. Ao diminuir esse threshold, estaremos cometendo, com mais frequência, o erro de classificar como canceladas reservas que, de fato, não seriam canceladas. O impacto desse erro para o negócio é que o hotel oferecerá, com maior frequência, algum tipo de vantagem para clientes que não iriam cancelar suas reservas.

Para avaliar o trade-off e escolhermos o melhor threshold, plotaremos abaixo o precision, recall e f1-score em função do threshold para o conjunto de validação.

<img src='threshold2.png' width="400">

Como previsto, à medida que o recall aumenta, a precision diminui. Como estamos dando mais importância ao recall, conforme explicado acima, vou escolher o threshold de 0.25. Em outras palavras, toda vez que nosso modelo calcular que a probabilidade de uma reserva ser cancelada é maior que 25%, nossa previsão será de que aquela reserva será cancelada.

O threshold de 25% dá, no conjunto de validação, um recall de 92.12% e uma precision de 77.33%.

Isso significa que, em nosso conjunto de validação, sempre que uma reserva vai ser cancelada pelo cliente, nosso modelo consegue prever corretamente em 92.12% dos casos. Em contrapartida, de todas as vezes que nosso modelo diz que uma reserva será cancelada, ele acerta em 77.33% das vezes. Ou seja, em 22.66% das vezes que nosso modelo diz que uma reserva será cancelada, na verdade ela não é.

## Estimando a acurácia do modelo em produção
Para estimar o desempenho do nosso modelo em produção, vamos utilizar o conjunto de teste. Perceba que essa é a primeira vez que utilizamos o conjunto de teste na modelagem. Isso garante que nenhum parâmetro ou hiperparâmetro foi selecionado para otimizar o modelo para esse conjunto.

Os pontos que temos no conjunto de teste nunca foram vistos pelo modelo, de modo que seu desempenho nesse conjunto é uma medida mais acurada do desempenho que o modelo terá no mundo real, em produção.

O modelo teve um desempenho no conjunto de teste muito próximo daquele observado no conjunto de validação (recall de 92,02% no teste, contra 92,12% na validação). Isso significa que nosso modelo generaliza bem para pontos inéditos, e nos dá maior confiança de usá-lo em produção.

A estimativa final é que nosso consegue prever corretamente 92.02% das reservas que serão canceladas. Como o precision encontrado foi de 77.44%, então em 22.56% das vezes que nosso modelo diz que uma reserva será cancelada, ele erra.

## Interpretação do modelo
Para explicar quais são as decisões que o modelo toma para chegar às previsões, utilizei os valores SHAP, com a implementação da biblioteca [shap](https://github.com/slundberg/shap). Uma explicação sobre o que é o SHAP e a análise completa da explicabilidade de nosso modelo pode ser vista no arquivo [explainability.ipynb](explainability.ipynb).

#### Importância de cada feature na previsão
No gráfico abaixo são mostradas a importância que cada feature tem para a previsão que nosso modelo faz. Podemos ver que o fato da reserva ter sido feita sem depósito é a feature mais importante para nosso modelo chegar à previsão. Em seguida vem a feature que diz se o cliente é de Portugal. A terceira feature mais importante é a agência que vendeu a reserva. Por fim, temos o número total de pedidos especiais feitos na reserva, e o LeadTime (tempo entre o registro da reserva e a data da estadia).

<img src='feature_importances.png' width="500">

Apesar das 5 features indicadas acima serem as mais importantes, vemos no gráfico abaixo que as outras features também influenciam de forma significativa o valor de SHAP de uma reserva e, consequentemente, a probabilidade de cancelamento calculada por nosso modelo.

#### Explicando as decisões do modelo
O gráfico abaixo dá uma visão geral das decisões que nosso modelo faz para chegar à previsão de um usuário. Cada linha representa uma feature diferente e deve ser lida separadamente.

<img src='summary_plot.png'>


Listamos abaixo alguns dos insights obtidos nessa etapa do projeto, incluindo a análise do gráfico acima. A análise completa (e os respectivos gráficos) pode ser vista no arquivo [explainability.ipynb](explainability.ipynb).

* Quando há um depósito com a reserva, nosso modelo diminui sutilmente a probabilidade de cancelamento. Em contrapartida, quando não houve depósito, a probabilidade de cancelamento aumenta drasticamente. Essa informação pode ser usada pelo hotel no seu business para evitar cancelamentos. Seria interessante estudar, por exemplo, se valeria a pena oferecer um desconto na diária do hotel, ou alguma outra vantagem, para reservas feitas com depósito.
* Quando o cliente é português, a probabilidade de cancelamento da reserva tende a aumentar. Mas existem algumas Agências que neutralizam a influência de o cliente ser português sobre a previsão do modelo.
* Existem agências que estão associadas a uma maior probabilidade de cancelamento.
* Reservas que não possuem pedidos especiais tendem a ter maior probabilidade de cancelamento. Quando há um ou dois pedidos especiais, essa probabilidade diminui. Quando temos três, quatro ou cinco pedidos especiais, diminui ainda mais. Esse efeito é reforçado caso o cliente seja do segmento de mercado Online TA.
* Quando temos um lead time de pouquíssimos dias, entre 0 e 5, a probabilidade de cancelamento é drasticamente diminuída. Para lead times maiores, entre 15 e 100 dias, o modelo não altera de forma consistente a previsão de probabilidade. A partir de 100 dias, um aumento no lead time tende a aumentar a previsão da probabilidade de cancelamento. Esse efeito é intensificado caso o cliente seja do segmento de mercado Online TA.
* Quando um cliente nunca cancelou uma reserva antes, isso quase não afeta a probabilidade prevista pelo modelo. Mas caso ele já o tenha feito, a probabilidade de cancelamento da reserva aumenta muito.
* Se um cliente não requisitou vaga de estacionamento, a previsão do modelo quase não sofre alteração. Mas no caso da vaga de estacionamento ter sido requisitada, a probabilidade de cancelamento cai drasticamente.

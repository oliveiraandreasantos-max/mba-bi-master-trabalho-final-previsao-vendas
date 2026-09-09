# Previsão de Vendas no Varejo com Machine Learning e Assistente Conversacional

#### Aluno: [Andréa Santos](https://github.com/oliveiraandreasantos-max)
#### Orientadora: [Manoela Kohler]

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

- [01 - EDA e tratamento de dados](01_EDA_e_tratamento.ipynb)
- [02 - Feature engineering](02_feature_engineering.ipynb)
- [03 - Modelagem](03_modelagem.ipynb)

---

### Resumo

Este trabalho desenvolve um sistema de previsão de vendas diárias para uma rede de varejo, combinando modelos de machine learning com um assistente conversacional que permite consultar os resultados em linguagem natural.

A base utilizada é a do Rossmann Store Sales, rede alemã de drogarias, com 844.338 registros diários de 1.115 lojas entre janeiro de 2013 e julho de 2015. A ela foram integradas quatro séries macroeconômicas mensais da economia alemã, obtidas do FRED: índice de confiança do consumidor, índice de preços ao consumidor, volume de vendas no varejo e taxa de desemprego.

Dois algoritmos foram comparados, Random Forest e XGBoost, cada um treinado em duas versões: na escala original das vendas e na escala logarítmica. O horizonte de previsão é de seis semanas, com validação por divisão temporal — treino até 18 de junho de 2015 e teste nas seis semanas seguintes.

A decisão metodológica central foi **excluir variáveis de defasagem (lags) e médias móveis**. Em um horizonte de seis semanas, esses valores não estão disponíveis no momento em que a previsão é feita, e utilizá-los produziria um modelo com desempenho artificialmente elevado e sem utilidade operacional. Em seu lugar, o patamar de vendas de cada loja entra por médias históricas calculadas exclusivamente sobre o período de treino.

O melhor modelo, XGBoost treinado na escala logarítmica, alcançou RMSPE de 12,51% contra 23,32% de um baseline construído com a média histórica de cada loja por dia da semana — uma redução de 46,4% no erro. O modelo não apresenta sobreajuste e mantém desempenho estável ao longo de todo o horizonte de seis semanas.

### Abstract

This work develops a daily sales forecasting system for a retail chain, combining machine learning models with a conversational assistant that allows results to be queried in natural language.

The dataset is Rossmann Store Sales, a German drugstore chain, comprising 844,338 daily records from 1,115 stores between January 2013 and July 2015. Four monthly macroeconomic series for the German economy were integrated from FRED: consumer confidence index, consumer price index, retail trade volume and unemployment rate.

Two algorithms were compared, Random Forest and XGBoost, each trained in two versions: on the original sales scale and on the logarithmic scale. The forecast horizon is six weeks, validated through a temporal split — training up to 18 June 2015 and testing over the following six weeks.

The central methodological decision was to **exclude lag and moving average features**. Over a six-week horizon these values are not available at prediction time, and using them would produce a model with artificially high measured performance and no operational value. Store sales levels instead enter through historical averages computed exclusively over the training period.

The best model, XGBoost trained on the logarithmic scale, achieved an RMSPE of 12.51% against 23.32% for a baseline built from each store's historical average by day of week — a 46.4% error reduction. The model shows no overfitting and maintains stable performance across the entire six-week horizon.

### 1. Introdução

Previsão de vendas é um problema central no varejo. Erros de previsão se traduzem diretamente em ruptura de estoque ou em capital imobilizado, e o efeito se propaga por toda a cadeia de suprimentos.

O problema tem duas características que dificultam a modelagem. A primeira é a heterogeneidade entre pontos de venda: lojas de perfis, portes e localizações diferentes respondem de forma distinta aos mesmos estímulos. A segunda é a multiplicidade de fatores simultâneos — calendário, promoções, feriados, concorrência e conjuntura econômica agem ao mesmo tempo sobre a demanda.

Este trabalho aborda o problema com modelos de árvore, que lidam bem com relações não lineares e com variáveis de naturezas distintas sem exigir normalização prévia. A escolha traz uma restrição relevante: Random Forest e XGBoost não são modelos nativos de série temporal e não enxergam a ordem cronológica dos registros. Toda informação sobre o tempo precisa ser fornecida explicitamente como coluna, e é essa exigência que estrutura a engenharia de variáveis do trabalho.

Um segundo componente do projeto é um dashboard com assistente conversacional baseado em LLM, que permite formular perguntas sobre a previsão em linguagem natural. O objetivo é aproximar o resultado do modelo de usuários de negócio que não trabalham diretamente com os dados.

#### 1.1 Objetivos

1. Construir um modelo de previsão de vendas diárias por loja com horizonte de seis semanas.
2. Comparar dois algoritmos de machine learning quanto a acurácia e custo computacional.
3. Identificar os indicadores mais relevantes para a previsão.
4. Avaliar a contribuição de variáveis macroeconômicas ao poder preditivo.
5. Disponibilizar os resultados em um dashboard com assistente conversacional.

#### 1.2 Base de dados

| | |
|---|---|
| Fonte principal | [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data) (Kaggle) |
| Período | 01/01/2013 a 31/07/2015 |
| Registros | 844.338, após remoção de dias com loja fechada |
| Lojas | 1.115 |
| Fontes complementares | Quatro séries mensais do [FRED](https://fred.stlouisfed.org/) para a economia alemã |

As séries macroeconômicas utilizadas são `CSCICP02DEM460S` (confiança do consumidor), `DEUCPIALLMINMEI` (índice de preços ao consumidor), `DEUSARTMISMEI` (volume de vendas no varejo, com ajuste sazonal) e `LMUNRRTTDEM156S` (taxa de desemprego).

### 2. Modelagem

#### 2.1 Tratamento dos dados

Dias com a loja fechada foram removidos: nesses dias a venda é zero por regra de funcionamento, não por comportamento de demanda, e mantê-los ensinaria o modelo a prever o calendário de operação em vez de prever vendas.

A variável `Customers` foi descartada por constituir vazamento de dados — o número de clientes de um dia só é conhecido após as vendas ocorrerem, e não estaria disponível no momento da previsão.

Cerca de 180 lojas apresentam interrupção na série entre julho e dezembro de 2014, por fechamento para reforma. Essas lojas foram marcadas pela variável `LojaComGap`, calculada sobre a base original antes do filtro de dias fechados — a contagem precisa distinguir linha ausente de dia sem funcionamento.

Lojas sem informação de distância do concorrente receberam a mediana da coluna, acompanhada da flag `SemInfoConcorrente`, que identifica os registros imputados.

#### 2.2 Engenharia de variáveis

Trinta variáveis explicativas, distribuídas em cinco blocos:

| Bloco | Variáveis | Exemplos |
|---|---|---|
| Temporais | 9 | ano, mês, dia, dia do ano, semana do ano, trimestre, dia da semana, início e fim de mês |
| Promoções e feriados | 6 | promoção do dia, participação e mês ativo do Promo2, feriado estadual, feriado escolar |
| Cadastro da loja | 8 | tipo, sortimento, distância do concorrente em escala logarítmica, tempo de concorrência, flags de dado ausente |
| Médias históricas | 3 | patamar da loja, patamar por dia da semana, patamar com e sem promoção |
| Macroeconômicas | 4 | variação anual de inflação, varejo e desemprego; diferença anual da confiança do consumidor |

**Exclusão de lags e médias móveis.** Variáveis de defasagem são, em geral, as mais preditivas em problemas de vendas diárias. Foram deliberadamente excluídas deste trabalho porque, em um horizonte de seis semanas, a venda de sete dias antes da data-alvo ainda não ocorreu quando a previsão é gerada. Incluí-las significaria treinar e avaliar o modelo com informação indisponível em produção — o erro metodológico mais comum na aplicação de modelos de árvore a problemas de previsão.

**Médias históricas em seu lugar.** Cada loja recebe três valores fixos, calculados uma única vez sobre o período de treino: venda média da loja, venda média por dia da semana e venda média com e sem promoção. Diferentemente dos lags, não dependem de dado recente e estão disponíveis em qualquer horizonte. O cálculo restrito ao treino é verificado automaticamente no notebook: as médias devem ser constantes por loja ao longo de todo o período, inclusive no conjunto de teste.

**Defasagem das variáveis macroeconômicas.** Um indicador mensal referente a junho não está disponível em junho — é publicado semanas depois. O dado do mês M é atribuído ao mês M+1 das vendas, evitando a mesma classe de vazamento dos lags.

**Transformação das séries macroeconômicas.** As séries entram como variação percentual em 12 meses, e não em nível: índices com tendência assumem no teste valores nunca vistos no treino, e modelos de árvore não extrapolam. O índice de confiança do consumidor foi tratado de forma distinta. A série oscila entre −2,0 e +0,6 no período e cruza o zero em abril de 2015, o que produz variações percentuais de até −300% — sem qualquer significado econômico. Para essa série usa-se a diferença absoluta em 12 meses, forma como indicadores de sentimento são convencionalmente reportados.

#### 2.3 Divisão treino/teste

Divisão temporal, com corte em 19 de junho de 2015. Divisão aleatória seria inadequada: embaralhar as linhas significaria treinar com dias posteriores aos que se pretende prever.

| | Registros | Período | Lojas |
|---|---|---|---|
| Treino | 802.942 | 01/01/2013 a 18/06/2015 | 1.115 |
| Teste | 41.396 | 19/06/2015 a 31/07/2015 | 1.115 |

O conjunto de teste corresponde a 4,9% dos registros, proporção equivalente ao desenho original da competição Rossmann. Todas as lojas estão presentes nos dois conjuntos. A venda média difere em 0,6% entre treino e teste, e as medidas centrais são praticamente idênticas, o que indica um período de teste representativo.

#### 2.4 Métricas

Quatro métricas foram calculadas. RMSE e MAE medem erro em euros; MAPE e RMSPE medem erro relativo. O **RMSPE** é a métrica oficial da competição Rossmann e orienta a comparação final, por permitir situar o resultado frente à literatura da competição.

#### 2.5 Escala logarítmica

Random Forest e XGBoost minimizam o erro absoluto ao quadrado. Uma loja que vende 15.000 euros por dia contribui muito mais para essa soma do que uma que vende 3.000, de modo que o modelo dedica sua capacidade às lojas maiores. O RMSPE, por outro lado, mede erro relativo: um erro de 500 euros vale 3,3% na loja grande e 16,7% na pequena. O modelo é treinado para um objetivo e avaliado por outro.

Treinar sobre o logaritmo das vendas alinha os dois. Como a diferença entre logaritmos corresponde a uma razão, minimizar erro absoluto no espaço logarítmico equivale aproximadamente a minimizar erro percentual no espaço original. A previsão é revertida à escala de euros antes do cálculo de qualquer métrica.

#### 2.6 Baseline

Como referência mínima, foi calculado o erro de prever simplesmente a média histórica de cada loja no dia da semana correspondente, sem modelo algum: RMSPE de 23,32%. Todo ganho reportado a seguir é medido sobre esse valor.

### 3. Resultados

#### 3.1 Comparação dos modelos

| Modelo | RMSE | MAE | MAPE | RMSPE | Ganho sobre baseline |
|---|---|---|---|---|---|
| Baseline (média histórica) | 1.649,65 | 1.240,35 | 18,27% | 23,32% | — |
| Random Forest (escala original) | 1.046,93 | 715,34 | 10,66% | 14,75% | 36,8% |
| Random Forest (escala log) | 1.042,92 | 708,35 | 10,44% | 14,37% | 38,4% |
| XGBoost (escala original) | 927,19 | 643,01 | 9,78% | 13,62% | 41,6% |
| **XGBoost (escala log)** | **892,70** | **610,81** | **9,10%** | **12,51%** | **46,4%** |

Os dois efeitos são separáveis e independentes. A troca de Random Forest por XGBoost rende cerca de 1,1 a 1,9 ponto percentual de RMSPE; a escala logarítmica rende de 0,4 a 1,1 ponto adicional, e beneficiou ambos os algoritmos.

A diferença de custo computacional também é relevante. O XGBoost treina em cerca de um minuto contra vários minutos do Random Forest, diferença que decorre da arquitetura: o método `hist` agrupa os valores em faixas antes de procurar os pontos de corte, em vez de testar cada valor único.

#### 3.2 Ausência de sobreajuste

| Modelo | RMSPE treino | RMSPE teste | Distância |
|---|---|---|---|
| Random Forest | 19,04% | 14,75% | −4,29 pp |
| Random Forest (log) | 17,96% | 14,37% | −3,60 pp |
| XGBoost | 17,24% | 13,62% | −3,62 pp |
| XGBoost (log) | 13,30% | 12,51% | −0,79 pp |

O erro no conjunto de teste é **menor** que no treino em todos os modelos, invertendo o padrão habitual. A explicação está na composição dos períodos: o treino cobre dois anos e meio, incluindo os fechamentos para reforma de 2014 e registros operacionais atípicos, enquanto o teste corresponde a seis semanas de um período regular. Não há qualquer indício de sobreajuste.

O modelo em escala logarítmica é também o mais consistente entre os dois períodos, com distância de apenas 0,79 ponto.

#### 3.3 Viés

| Modelo | Viés |
|---|---|
| Random Forest | +0,82% |
| Random Forest (log) | −0,16% |
| XGBoost | +1,71% |
| XGBoost (log) | +0,47% |

Os modelos em escala original superestimam sistematicamente as vendas. A transformação logarítmica reduz esse viés em ambos os algoritmos. O resultado contraria a expectativa teórica: a reversão da transformação logarítmica introduz, em geral, um viés para baixo, decorrente de a média dos logaritmos não corresponder ao logaritmo da média. Neste caso, a assimetria da distribuição original predominou sobre esse efeito, e a transformação corrigiu um viés em vez de introduzi-lo.

#### 3.4 Estabilidade ao longo do horizonte

| Semana do horizonte | Erro percentual médio | Registros |
|---|---|---|
| 1 | 8,14% | 6.716 |
| 2 | 10,48% | 6.721 |
| 3 | 10,82% | 6.717 |
| 4 | 7,57% | 6.710 |
| 5 | 8,69% | 6.709 |
| 6 | 9,02% | 6.710 |

O erro oscila entre 7,6% e 10,8% sem tendência de crescimento. O modelo não degrada conforme a previsão se afasta do último dado conhecido, o que sustenta o uso da previsão em toda a janela de seis semanas.

#### 3.5 Indicadores mais relevantes

Considerando todas as variáveis, as três médias históricas concentram 75% da importância. Isso é esperado por construção: elas resumem o patamar de vendas de cada loja, principal fator de variação da base. Para responder ao objetivo do trabalho, a leitura relevante é a ordenação das demais variáveis entre si.

| Variável | Peso relativo |
|---|---|
| Promo | 32,71% |
| DayOfWeek | 10,41% |
| SemanaDoAno | 8,47% |
| Dia | 7,30% |
| DiaDoAno | 7,08% |
| Desemprego (variação anual) | 3,46% |
| Mes | 3,15% |
| Inflação (variação anual) | 2,97% |
| StateHoliday | 2,45% |
| Trimestre | 2,25% |
| Confiança do consumidor (diferença anual) | 2,15% |

**Promoção é o fator acionável dominante**, com peso três vezes maior que o segundo colocado. Somada à variável `MediaLojaPromo`, a mais importante do conjunto completo, o resultado sustenta a recomendação de tratar a sensibilidade promocional loja a loja, e não como parâmetro único da rede.

**O calendário responde por cerca de 40%** das demais variáveis, distribuído entre dia da semana, semana do ano, dia do mês e dia do ano.

**As variáveis macroeconômicas somam aproximadamente 10%.** O valor é modesto, mas não nulo, e deve ser lido à luz da limitação descrita na seção 4.

**A posição competitiva é irrelevante.** Distância do concorrente, tempo de concorrência e as flags associadas somam menos de 4%. No período analisado, a variação de vendas é explicada por calendário e promoção, não por posição competitiva.

#### 3.6 Erro por grupo

| Grupo | Erro médio | Registros |
|---|---|---|
| Lojas sem interrupção na série | 9,00% | 34.713 |
| Lojas com interrupção (reforma em 2014) | 9,62% | 6.683 |
| Dias sem promoção | 9,21% | 23.581 |
| Dias com promoção | 8,95% | 17.815 |

A diferença entre lojas com e sem interrupção é de 0,6 ponto percentual — real, mas modesta. Considerou-se a criação de uma variável marcando os dias posteriores à reabertura, para capturar o efeito de reinauguração; a diferença observada não justifica o acréscimo de complexidade.

#### 3.7 Importância não é contribuição

A variável `Ano` responde por apenas 0,25% da importância média, mas sua remoção eleva o RMSPE de 12,51% para 13,06%.

Para verificar se essa diferença é significativa, mediu-se antes a variação natural do modelo: três treinos idênticos com sementes aleatórias distintas produziram RMSPE de 12,51%, 12,66% e 12,66%, uma amplitude de 0,15 ponto. A perda causada pela remoção de `Ano` é quase quatro vezes maior que esse ruído de fundo, o que caracteriza contribuição real.

O aparente paradoxo tem explicação. Importância de variáveis mede com que frequência a variável é **utilizada** nas divisões das árvores; o teste de remoção mede quanto ela **contribui** para o resultado. `Ano` é raramente utilizada, mas nas ocasiões em que o é, decide algo que nenhuma outra variável substitui — provavelmente a tendência de crescimento entre os anos, informação que as médias históricas não carregam por serem calculadas sobre todo o período.

### 4. Limitações

**Janela de teste curta para avaliar variáveis macroeconômicas.** O conjunto de teste cobre seis semanas, ou um a dois meses de calendário. Nesse intervalo, indicadores mensais são praticamente constantes — a taxa de desemprego assume apenas 12 valores distintos em toda a base. Não é possível validar o poder preditivo dessas variáveis com este desenho. Sua contribuição ao trabalho é sustentar a análise de cenário econômico e alimentar o assistente do dashboard.

**Ausência de dinâmicas de curto prazo.** Ao excluir lags e médias móveis, o modelo não captura tendências recentes de uma loja específica nem o efeito residual de uma promoção próxima. A escolha privilegia a validade operacional da previsão em horizontes longos sobre a acurácia de curto prazo.

**A variável `Ano` não se sustenta fora da amostra.** Assume apenas três valores e o conjunto de teste está inteiramente em 2015. Modelos de árvore não extrapolam, de modo que o modelo treinado não serviria para prever um ano que não observou sem retreino.

**Defasagem única para todas as séries macroeconômicas.** O dado de varejo alemão costuma ser publicado próximo ao fim do mês seguinte ao de referência; rigorosamente, uma defasagem de dois meses seria mais adequada para essa série específica. Optou-se por defasagem única por simplicidade.

**Validação por divisão temporal única.** Validação cruzada temporal com janelas deslizantes forneceria estimativa mais robusta, ao custo de multiplicar o tempo de treino.

**Imputação com estatística global.** A mediana de `CompetitionDistance` foi calculada sobre a base completa. Como a distância é atributo fixo da loja e não varia no tempo, o efeito é desprezível.

**Assistente sem fundamentação em dados externos.** O assistente conversacional do dashboard recebe a saída do modelo diretamente no prompt e responde questões econômicas gerais a partir de seu próprio conhecimento, sem consulta a fontes externas atualizadas.

### 5. Conclusões

O modelo final, XGBoost treinado na escala logarítmica das vendas, reduz o erro de previsão em 46,4% frente a um baseline de média histórica, alcançando RMSPE de 12,51% em um horizonte de seis semanas.

O resultado deve ser lido junto com a restrição metodológica que o produziu. As soluções mais bem colocadas na competição original alcançaram valores próximos de 10% de RMSPE, com engenharia de variáveis intensiva baseada em defasagens. Este trabalho abriu mão dessas variáveis por decisão deliberada, uma vez que não estariam disponíveis no momento da previsão em uso real. A diferença de aproximadamente dois pontos percentuais é o custo dessa escolha, e o ganho é um modelo cuja avaliação corresponde ao desempenho que se obteria em operação.

Quanto aos indicadores mais relevantes, promoção é o fator acionável dominante, com peso três vezes superior ao segundo colocado, e sua eficácia varia entre lojas. O calendário responde por parcela expressiva do restante. Posição competitiva mostrou-se irrelevante no período analisado.

A comparação entre algoritmos favorece o XGBoost nos dois critérios avaliados: acurácia superior e custo computacional substancialmente menor. A transformação logarítmica do alvo mostrou-se benéfica para ambos os algoritmos, melhorando simultaneamente acurácia, estabilidade entre treino e teste, e viés.

Três achados metodológicos merecem registro. O erro de teste inferior ao de treino, decorrente da composição dos períodos, evidencia ausência de sobreajuste. O viés da transformação logarítmica ocorreu em direção contrária à expectativa teórica. E a variável `Ano` demonstra que importância de variáveis e contribuição efetiva são medidas distintas, o que recomenda cautela na interpretação de rankings de importância.

#### 5.1 Trabalhos futuros

- Validação cruzada temporal com janelas deslizantes.
- Comparação entre a formulação com e sem defasagens, mantendo o horizonte fixo, para quantificar o custo exato da restrição adotada.
- Teste de defasagem de dois meses para as séries macroeconômicas, como análise de sensibilidade.
- Avaliação sobre uma janela de teste mais longa, que permita medir a contribuição real das variáveis macroeconômicas.
- Submissão ao conjunto oficial da competição para validação externa independente.

---

Matrícula: [241.100.405]

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*

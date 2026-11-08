#!/usr/bin/env python3
"""Gera o corpus FICTÍCIO usado pela avaliação de RAG (evals/fixtures/).

Todo o conteúdo é inventado: pessoas, empresas, endereços, valores e códigos
não existem. Os arquivos gerados são versionados; rode de novo só se mudar o
texto aqui (o PDF e o DOCX saem iguais em conteúdo, mas não byte a byte).

    .venv/Scripts/python.exe evals/make_fixtures.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# --------------------------------------------------------------------- textos

REUNIAO_FALCON = """# Reunião do projeto Falcon — 14/10/2026

Participantes: Júlia Mendes (produto), Rafael Okada (engenharia), Bianca Torres (design) e Caio Lemos (financeiro).

## Contexto

O Falcon é o aplicativo interno de reservas de salas da Vértice Consultoria. A versão atual trava quando duas pessoas reservam a mesma sala no mesmo minuto, e o time de facilities reclamou três vezes no último mês.

## Decisões

1. O lançamento da versão 2.0 foi adiado de 3 de novembro para 24 de novembro, porque a migração do banco de dados ainda não passou nos testes de carga.
2. A reserva passa a usar bloqueio otimista: a segunda pessoa vê a mensagem "sala acabou de ser reservada" em vez de um erro genérico.
3. O orçamento aprovado para a fase 2 é de R$ 48.000, dos quais R$ 12.000 ficam reservados para testes com usuários.

## Responsáveis

Rafael cuida da migração do banco e do teste de carga com 500 reservas simultâneas. Bianca entrega o novo fluxo de cancelamento até 4 de novembro. Caio acompanha o gasto semanal e avisa se passar de 80% do orçamento.

## Riscos

O maior risco é a integração com o calendário corporativo, que muda de fornecedor em dezembro. Se a troca atrasar, o Falcon continua lendo o calendário antigo por mais um trimestre.

## Próxima reunião

Sexta-feira, 31 de outubro, às 14h, na sala Aurora do terceiro andar.
"""

REGULAMENTO_CONDOMINIO = """REGULAMENTO INTERNO — CONDOMÍNIO RESIDENCIAL JARDIM DAS ACÁCIAS (documento fictício)

1. SILÊNCIO
O horário de silêncio vai das 22h às 7h nos dias úteis e das 23h às 9h aos sábados, domingos e feriados. Obras barulhentas só podem acontecer de segunda a sexta, das 9h às 17h.

2. PISCINA
A piscina funciona de terça a domingo, das 8h às 21h. Às segundas ela fica fechada para limpeza. Cada apartamento pode levar no máximo dois convidados por vez, e crianças menores de 12 anos precisam estar acompanhadas de um adulto.

3. MUDANÇAS
Mudanças devem ser agendadas na portaria com pelo menos 48 horas de antecedência e só acontecem de segunda a sábado, das 8h às 18h. O elevador de serviço é forrado pela administração antes de cada mudança.

4. ANIMAIS
Animais de estimação são permitidos, mas devem circular no colo ou na guia nas áreas comuns. Cães de grande porte usam apenas o elevador de serviço.

5. SALÃO DE FESTAS
O salão de festas é reservado pelo aplicativo do condomínio, com taxa de uso de R$ 180 por evento, e a festa deve terminar até a meia-noite.

6. ASSEMBLEIA E ORÇAMENTO
Na assembleia de setembro de 2026 foi aprovada uma taxa extra de R$ 95 por mês, durante dez meses, para trocar a impermeabilização da cobertura. O orçamento anual do condomínio para 2027 ficou em R$ 612.000.

7. MULTAS
O descumprimento deste regulamento gera advertência por escrito na primeira vez e multa de meia taxa condominial a partir da segunda ocorrência no mesmo ano.
"""

RECEITAS = """# Receitas da família

## Pão de queijo da vó Lurdes

Ingredientes: 500 g de polvilho azedo, 250 ml de leite, 120 ml de óleo, 2 ovos, 200 g de queijo minas meia cura ralado e 1 colher de chá de sal.

Ferva o leite com o óleo e o sal e escalde o polvilho, mexendo até esfriar. Junte os ovos um a um e depois o queijo. Faça bolinhas do tamanho de uma noz e asse em forno preaquecido a 200 °C por cerca de 25 minutos, até dourar.

## Bolo de cenoura com cobertura de chocolate

Bata no liquidificador 3 cenouras médias, 4 ovos e 240 ml de óleo. Misture com 2 xícaras de açúcar, 2 xícaras e meia de farinha de trigo e 1 colher de sopa de fermento. Asse a 180 °C por 40 minutos numa forma untada.

Para a cobertura, leve ao fogo 1 colher de manteiga, 3 colheres de chocolate em pó, 1 xícara de açúcar e meia xícara de leite, mexendo até engrossar.

## Feijão tropeiro rápido

Refogue 200 g de bacon em cubos, junte 2 dentes de alho, meia cebola e 3 xícaras de feijão carioca cozido e escorrido. Acrescente farinha de mandioca aos poucos, 2 ovos mexidos e cheiro-verde no final.
"""

VIAGEM_LISBOA = """# Viagem para Lisboa — março de 2027 (roteiro fictício)

## Voos

Ida no dia 6 de março de 2027, voo TP 0088, saindo de Guarulhos às 22h40 e chegando em Lisboa às 12h10 do dia seguinte. Volta no dia 16 de março, voo TP 0087, às 10h05. O código da reserva é QX7LMB.

## Hospedagem

Hotel Miradouro Azul, na Rua das Gaivotas, 18, no bairro de Alfama. Check-in a partir das 15h. O café da manhã está incluído e a reserva pode ser cancelada sem custo até 20 de fevereiro.

## Roteiro

Dia 8: Mosteiro dos Jerónimos e pastéis de Belém pela manhã; à tarde, Torre de Belém.

Dia 9: bate-volta para Sintra, com o Palácio da Pena. Comprar os ingressos com antecedência pelo site, porque esgotam.

Dia 10: Castelo de São Jorge e jantar com fado em Alfama.

Dias 11 e 12: Porto de trem, saindo da estação de Santa Apolónia às 8h39.

## Orçamento

Separamos 2.400 euros para a viagem toda, sem contar as passagens. O seguro-viagem foi contratado com cobertura de 30 mil euros, exigida para entrar no espaço Schengen.
"""

NOTAS_CALCULO = """# Cálculo I — anotações de aula (turma B)

## Limites

Um limite descreve o valor de que uma função se aproxima quando x tende a um ponto. O limite fundamental trigonométrico diz que sen(x)/x tende a 1 quando x tende a 0. Quando aparece uma indeterminação do tipo 0/0, tente fatorar ou racionalizar antes de qualquer coisa.

## Derivadas

A derivada mede a taxa de variação instantânea. Regras para decorar: a derivada de x elevado a n é n vezes x elevado a n menos 1; a regra do produto é f'g + fg'; a regra da cadeia multiplica a derivada de fora pela derivada de dentro.

## Regra de L'Hôpital

Se o limite der 0/0 ou infinito sobre infinito, pode-se derivar numerador e denominador separadamente e calcular o limite de novo. Não vale para outros tipos de indeterminação sem antes reescrever a expressão.

## Avaliações

A primeira prova (P1) é no dia 19 de novembro de 2026, às 10h, na sala 204 do bloco C, e cobre limites e derivadas. A lista de exercícios 3 vale 1 ponto extra e deve ser entregue até 12 de novembro. O professor Henrique Saldanha atende dúvidas às quartas, das 16h às 18h.
"""

MANUAL_ROTEADOR = """MANUAL RÁPIDO — ROTEADOR NIMBUS AX1800 (produto fictício)

PRIMEIRO ACESSO
Conecte o cabo de rede do modem na porta azul (WAN). Acesse http://192.168.10.1 no navegador. O usuário padrão é admin e a senha padrão vem impressa na etiqueta embaixo do aparelho.

REDE WI-FI
O roteador cria duas redes: Nimbus-2G, de maior alcance, e Nimbus-5G, mais rápida e com menos interferência. Para trocar a senha do Wi-Fi, entre em Configurações > Sem fio > Segurança e escolha WPA3.

RESET DE FÁBRICA
Com o roteador ligado, segure o botão RESET na parte de trás por 10 segundos, até a luz ficar laranja piscando. Todas as configurações voltam ao padrão de fábrica.

LUZES
Luz verde fixa: internet funcionando. Luz vermelha: sem conexão com o provedor. Luz laranja piscando: reiniciando ou atualizando o firmware — não desligue o aparelho nesse momento.

GARANTIA
A garantia é de 24 meses a partir da data da nota fiscal. O suporte técnico atende pelo telefone 0800 555 0199, de segunda a sábado, das 8h às 20h.
"""

PLANO_TREINO = """PLANO DE TREINO — PREPARAÇÃO PARA 10 KM (fictício)

OBJETIVO
Completar a corrida de 10 km do dia 14 de dezembro de 2026 em menos de 60 minutos, partindo de um ritmo atual de 6min40s por quilômetro.

SEMANA TÍPICA
Segunda: descanso ou alongamento leve.
Terça: intervalado — 6 tiros de 800 metros com 2 minutos de trote entre eles.
Quarta: musculação com foco em glúteos, posteriores e core.
Quinta: rodagem leve de 5 km em ritmo confortável.
Sábado: longão, aumentando 1 km por semana até chegar a 12 km.

ALIMENTAÇÃO
Antes do longão, comer uma banana com aveia uma hora antes. Depois do treino, priorizar proteína e repor líquidos.

CUIDADOS
Trocar o tênis a cada 600 km rodados. Se aparecer dor na canela por mais de três dias, reduzir o volume pela metade e procurar a fisioterapeuta Paula Reis.
"""

CONTRATO_PARAGRAFOS = [
    "CONTRATO DE LOCAÇÃO RESIDENCIAL (documento fictício para testes)",
    "LOCADOR: Otávio Brandão Siqueira. LOCATÁRIA: Júlia Mendes. IMÓVEL: apartamento 302 da "
    "Rua dos Ipês, 145, bairro Monte Verde, São Luís - MA.",
    "CLÁUSULA 1 - PRAZO. A locação vale por 30 meses, começando em 1º de agosto de 2026 e "
    "terminando em 31 de janeiro de 2029.",
    "CLÁUSULA 2 - VALOR E PAGAMENTO. O aluguel mensal é de R$ 2.350,00, pago até o dia 5 de "
    "cada mês por transferência para a conta indicada pelo locador.",
    "CLÁUSULA 3 - ATRASO. O pagamento depois do vencimento gera multa de 2% sobre o valor do "
    "aluguel, mais juros de 1% ao mês, calculados por dia de atraso.",
    "CLÁUSULA 4 - REAJUSTE. O aluguel será reajustado a cada 12 meses pela variação do IPCA. "
    "As partes abrem mão do IGP-M.",
    "CLÁUSULA 5 - GARANTIA. A locatária entrega caução equivalente a três aluguéis, devolvida "
    "corrigida ao fim do contrato se o imóvel estiver em boas condições.",
    "CLÁUSULA 6 - ANIMAIS E REFORMAS. É permitido ter animais de pequeno porte. Reformas "
    "estruturais dependem de autorização por escrito do locador.",
    "CLÁUSULA 7 - RESCISÃO. Se a locatária sair antes de 12 meses, paga multa proporcional "
    "de três aluguéis; depois de 12 meses, basta avisar com 30 dias de antecedência.",
    "CLÁUSULA 8 - CONDOMÍNIO E IPTU. A taxa de condomínio ordinária e o IPTU ficam por conta "
    "da locatária. Taxas extraordinárias, como obras na fachada ou na cobertura, são pagas "
    "pelo locador.",
    "CLÁUSULA 9 - CONTAS DE CONSUMO. As contas de energia, água e gás devem ser transferidas "
    "para o nome da locatária em até 15 dias depois da entrega das chaves.",
    "CLÁUSULA 10 - VISTORIA. As partes assinam um laudo de vistoria com fotos na entrega das "
    "chaves. Na saída, o imóvel deve ser devolvido pintado na mesma cor, branco gelo.",
    "CLÁUSULA 11 - MANUTENÇÃO. Pequenos reparos de uso, como troca de lâmpadas, torneiras e "
    "resistência do chuveiro, são da locatária. Problemas estruturais, infiltrações e a "
    "parte elétrica embutida são responsabilidade do locador, que tem 10 dias para resolver "
    "depois de avisado por escrito.",
    "CLÁUSULA 12 - SUBLOCAÇÃO. É proibido sublocar, ceder ou emprestar o imóvel, no todo ou em "
    "parte, inclusive por aplicativos de hospedagem por temporada.",
    "CLÁUSULA 13 - VENDA DO IMÓVEL. Se o locador decidir vender o apartamento, a locatária tem "
    "preferência na compra em igualdade de condições e deve responder em 30 dias.",
    "CLÁUSULA 14 - VAGA DE GARAGEM. O aluguel inclui a vaga de garagem número 27, coberta, no "
    "primeiro subsolo, e um depósito de 2 metros quadrados no térreo.",
    "CLÁUSULA 15 - FORO. Fica eleito o foro da comarca de São Luís para resolver qualquer "
    "dúvida sobre este contrato.",
]

RELATORIO_PARAGRAFOS = [
    "Relatório do 3º trimestre de 2026 — Ateliê Lume (empresa fictícia)",
    "Resumo: o Ateliê Lume faturou R$ 184.300 no terceiro trimestre, crescimento de 12% em "
    "relação ao segundo trimestre, puxado pelas vendas da linha de luminárias de papel.",
    "Vendas por canal: a loja online respondeu por 61% do faturamento, as feiras de design por "
    "27% e as vendas para arquitetos por 12%.",
    "Custos: o custo do frete subiu 18% depois da mudança de transportadora em agosto, o que "
    "reduziu a margem bruta de 46% para 41%.",
    "Clientes: a taxa de recompra ficou em 23% e o tempo médio de entrega caiu de 9 para 6 "
    "dias úteis.",
    "Equipe: o ateliê fechou o trimestre com 9 pessoas. Contratamos uma designer de produto em "
    "julho e uma pessoa para o atendimento em setembro. A rotatividade foi zero.",
    "Marketing: o investimento em anúncios foi de R$ 14.200. O custo de aquisição de cliente "
    "caiu para R$ 38, e a campanha com influenciadoras de decoração trouxe 1.150 pedidos "
    "novos. O e-mail marketing continua sendo o canal de maior conversão, com 4,1%.",
    "Estoque: terminamos setembro com 640 peças prontas. O papel japonês importado demorou 45 "
    "dias para chegar, e por isso a Luminária Origami ficou 11 dias sem estoque em agosto.",
    "Riscos: dependência de um único fornecedor de papel, aumento do frete no fim do ano e a "
    "possível alta do dólar, que encarece as lâmpadas de LED importadas.",
    "Próximos passos: negociar um contrato anual de frete até 15 de dezembro e lançar a "
    "coleção de verão em 10 de janeiro de 2027.",
]

RELATORIO_TABELA: List[Tuple[str, str, str]] = [
    ("Produto", "Unidades", "Receita (R$)"),
    ("Luminária Origami", "412", "86.520"),
    ("Abajur Duna", "198", "51.480"),
    ("Pendente Nuvem", "137", "46.300"),
]

# Continuações dos documentos acima (ficam no mesmo arquivo; separadas aqui
# só para facilitar a leitura).

REUNIAO_FALCON_EXTRA = """
## Detalhes técnicos discutidos

Rafael explicou que a migração troca o SQLite por PostgreSQL 16 e cria um índice único por sala e horário. O teste de carga da semana passada derrubou o servidor com 320 reservas por minuto; a meta é aguentar 500 sem erro e com tempo de resposta abaixo de 300 milissegundos.

Bianca mostrou o protótipo do novo cancelamento: a pessoa cancela com dois toques e a sala volta a aparecer como livre na hora. Também entra um lembrete por e-mail 15 minutos antes da reunião, que pode ser desligado nas preferências.

## Métricas de uso

Em setembro foram 4.870 reservas, 18% delas canceladas. A sala mais disputada é a Aurora, com taxa de ocupação de 92% entre 10h e 16h. Cerca de 37% das reservas são feitas pelo celular.

## Pendências da reunião anterior

A integração com o crachá para liberar a porta da sala continua parada esperando o time de segurança. O relatório mensal de ocupação para a diretoria foi entregue no dia 2 de outubro.
"""

REGULAMENTO_CONDOMINIO_EXTRA = """
8. GARAGEM
Cada apartamento tem direito a uma vaga demarcada. É proibido lavar carros na garagem e estacionar motos fora do bicicletário coberto do segundo subsolo. Visitantes estacionam apenas nas quatro vagas em frente à portaria, por no máximo 3 horas.

9. LIXO E RECICLAGEM
O lixo orgânico deve ser descartado no contêiner cinza do térreo até as 20h. Recicláveis vão separados nos contêineres coloridos ao lado do portão de serviço. Óleo de cozinha usado é recolhido em garrafas PET na primeira sexta-feira de cada mês.

10. ENTREGAS
Entregadores não sobem aos apartamentos. As encomendas ficam na sala de achados da portaria por até 15 dias, e o morador é avisado pelo aplicativo do condomínio.

11. ACADEMIA
A academia funciona todos os dias, das 6h às 22h, com uso de toalha obrigatório. Menores de 16 anos só podem usar os aparelhos acompanhados dos pais.
"""

RECEITAS_EXTRA = """
## Moqueca de peixe da tia Neide

Tempere 1 kg de pescada amarela em postas com limão, alho e sal por 30 minutos. Numa panela de barro, faça camadas de cebola, tomate e pimentão em rodelas, acomode o peixe e regue com 200 ml de leite de coco e 2 colheres de azeite de dendê. Cozinhe tampado em fogo médio por 20 minutos, sem mexer, e finalize com coentro.

## Arroz doce de festa junina

Cozinhe 1 xícara de arroz em 3 xícaras de água com casca de limão e cravo. Quando a água secar, junte 1 litro de leite, 1 lata de leite condensado e meia xícara de açúcar, mexendo em fogo baixo por 25 minutos até ficar cremoso. Polvilhe canela antes de servir.

## Dicas gerais

Polvilho azedo deixa o pão de queijo mais aerado; o doce deixa a massa mais densa. Bolo de cenoura fica mais fofo com os ovos em temperatura ambiente. Massa de pão de queijo congela bem por até três meses.
"""

VIAGEM_LISBOA_EXTRA = """
## Documentos e dinheiro

Levar passaporte com validade mínima de seis meses, comprovante do hotel, passagem de volta e o certificado do seguro-viagem impresso. O cartão de débito internacional já foi desbloqueado para uso em Portugal e na Espanha. Trocar só 200 euros em espécie; o resto no cartão.

## Transporte em Lisboa

Comprar o cartão Viva Viagem no aeroporto e carregar no modo zapping. O elétrico 28 passa por Alfama e pela Graça, mas lota cedo: ir antes das 9h. Para Belém, o elétrico 15 sai da Praça da Figueira.

## Restaurantes anotados

Taberna do Largo, em Alfama, para bacalhau à brás. Cervejaria Ramiro para frutos do mar, chegar antes das 19h para não pegar fila. Manteigaria, no Chiado, tem o pastel de nata preferido da Bianca.
"""

NOTAS_CALCULO_EXTRA = """
## Integrais (prévia para a P2)

A integral indefinida é a antiderivada: a integral de x elevado a n é x elevado a n mais 1, dividido por n mais 1, mais a constante C, para n diferente de -1. A integral definida representa a área sob a curva entre dois pontos, e o Teorema Fundamental do Cálculo liga as duas ideias.

## Dicas do monitor

O monitor Lucas sugere resolver os exercícios ímpares do capítulo 3 do livro, que têm resposta no final. Ele dá plantão na biblioteca às terças, das 14h às 16h. Na P1 do semestre passado, a questão mais errada foi a de regra da cadeia com funções trigonométricas.

## Critério de aprovação

A média final é (P1 + P2 + P3) dividido por 3, mais os pontos extras das listas. Quem tiver média igual ou maior que 7 passa direto; entre 4 e 7 faz exame final em 15 de dezembro.
"""

MANUAL_ROTEADOR_EXTRA = """
REDE DE CONVIDADOS
Em Configurações > Convidados, ative a rede Nimbus-Visitas. Ela tem senha própria, não enxerga os outros aparelhos da casa e pode ser desligada automaticamente depois de 4, 8 ou 24 horas.

CONTROLE DOS PAIS
Em Configurações > Família é possível bloquear sites por categoria e definir horários sem internet para cada aparelho, por exemplo desligar o tablet das crianças das 22h às 7h.

ATUALIZAÇÃO DE FIRMWARE
O roteador procura atualizações toda madrugada às 3h. Para atualizar manualmente, vá em Sistema > Firmware > Verificar agora. A atualização leva cerca de 5 minutos e reinicia o aparelho.

MESH
O Nimbus AX1800 pode ser combinado com até três satélites Nimbus Mesh para cobrir casas grandes. Pareie cada satélite apertando o botão WPS dos dois aparelhos em até 2 minutos.
"""

PLANO_TREINO_EXTRA = """
PROGRESSÃO DO LONGÃO
Semana 1: 7 km. Semana 2: 8 km. Semana 3: 9 km. Semana 4: semana de recuperação com 6 km. Semana 5: 10 km. Semana 6: 11 km. Semana 7: 12 km. Semana 8: polimento, só 6 km leves antes da prova.

ZONAS DE FREQUÊNCIA CARDÍACA
A frequência cardíaca máxima estimada é 186 batimentos por minuto. Rodagem leve fica na zona 2, entre 112 e 130 bpm. Os tiros de 800 metros podem chegar à zona 4, entre 149 e 167 bpm.

DIA DA PROVA
Acordar às 4h30, tomar café com pão, banana e mel até 5h30. A largada é às 6h30 no Parque da Lagoa. Retirar o kit com o número de peito na sexta-feira anterior, das 10h às 19h.
"""

# Documentos-distratores: parecidos com os de cima de propósito (mesmos
# temas, valores e datas diferentes), para a busca não acertar só pelo assunto.

REUNIAO_ORION = """# Reunião do projeto Órion — 9/9/2026

Participantes: Júlia Mendes, Rafael Okada e Marta Quintela (jurídico).

## Contexto

O Órion é o portal de contratos da Vértice Consultoria, onde os clientes assinam propostas e aditivos. O sistema atual exige impressão e digitalização, e a meta é ter assinatura eletrônica.

## Decisões

1. O piloto da assinatura eletrônica começa em 6 de outubro com três clientes.
2. O orçamento da fase 1 é de R$ 27.500, incluindo a licença anual do fornecedor de assinatura.
3. Documentos com mais de 50 páginas continuam no fluxo antigo até o fim do piloto.

## Responsáveis

Marta revisa os modelos de contrato até 22 de setembro. Rafael integra a API do fornecedor e cuida dos registros de auditoria exigidos pelo jurídico.

## Próxima reunião

Terça-feira, 23 de setembro, às 10h, por videochamada.
"""

ATA_ASSEMBLEIA_2025 = """ATA DA ASSEMBLEIA GERAL ORDINÁRIA — CONDOMÍNIO RESIDENCIAL JARDIM DAS ACÁCIAS — MARÇO DE 2025 (documento fictício)

Aos quinze dias de março de 2025, reuniram-se os condôminos no salão de festas, com presença de 41 das 64 unidades.

PAUTA 1 — PRESTAÇÃO DE CONTAS
As contas de 2024 foram aprovadas por unanimidade. O condomínio terminou o ano com saldo de R$ 38.700 no fundo de reserva.

PAUTA 2 — ORÇAMENTO DE 2025
O orçamento anual de 2025 foi aprovado em R$ 574.000, com reajuste de 6% na taxa condominial a partir de abril.

PAUTA 3 — PORTARIA
Foi aprovada a troca da portaria presencial noturna por portaria remota das 0h às 6h, com economia estimada de R$ 4.100 por mês.

PAUTA 4 — PISCINA
A proposta de aquecer a piscina foi rejeitada por 23 votos a 18, por causa do custo de energia.

PAUTA 5 — ELEIÇÃO
Foi eleita síndica a senhora Rosângela Prado, do apartamento 1104, para mandato de dois anos.
"""

VIAGEM_BUENOS_AIRES = """# Viagem para Buenos Aires — julho de 2025 (roteiro fictício, já realizada)

## Voos

Ida em 12 de julho de 2025, voo AR 1255, saindo de Guarulhos às 9h15. Volta em 19 de julho, às 18h40. Código da reserva: BZ4TRK.

## Hospedagem

Apartamento alugado em Palermo Soho, na Calle Honduras, com check-in às 14h.

## O que fizemos

Caminhada por San Telmo no domingo, com a feira de antiguidades. Show de tango no Café Tortoni. Bate-volta de barco para Colonia del Sacramento, no Uruguai, saindo do porto de Puerto Madero às 8h.

## Gastos

Gastamos cerca de 1.300 dólares na viagem, sem as passagens. Valeu a pena levar dólares em espécie para trocar nas casas de câmbio.
"""

NOTAS_FISICA = """# Física I — anotações de aula (turma A)

## Cinemática

Velocidade média é o deslocamento dividido pelo tempo. No movimento uniformemente variado, a posição é s = s0 + v0·t + a·t²/2, e a equação de Torricelli relaciona velocidades e deslocamento sem usar o tempo.

## Leis de Newton

A segunda lei diz que a força resultante é igual à massa vezes a aceleração. O peso é a massa vezes a gravidade, e a força normal nem sempre é igual ao peso, por exemplo num plano inclinado.

## Avaliações

A prova 1 de Física é no dia 26 de novembro de 2026, às 8h, no laboratório 3. A professora Clara Vasconcelos atende às segundas, das 10h às 12h. O relatório do experimento do pêndulo deve ser entregue até 20 de novembro.
"""

TEXT_FILES = {
    "reuniao_projeto_falcon.md": REUNIAO_FALCON + REUNIAO_FALCON_EXTRA,
    "regulamento_condominio.txt": REGULAMENTO_CONDOMINIO + REGULAMENTO_CONDOMINIO_EXTRA,
    "receitas_familia.md": RECEITAS + RECEITAS_EXTRA,
    "viagem_lisboa.md": VIAGEM_LISBOA + VIAGEM_LISBOA_EXTRA,
    "notas_aula_calculo.md": NOTAS_CALCULO + NOTAS_CALCULO_EXTRA,
    "manual_roteador.txt": MANUAL_ROTEADOR + MANUAL_ROTEADOR_EXTRA,
    "plano_treino.txt": PLANO_TREINO + PLANO_TREINO_EXTRA,
    "reuniao_projeto_orion.md": REUNIAO_ORION,
    "ata_assembleia_2025.txt": ATA_ASSEMBLEIA_2025,
    "viagem_buenos_aires_2025.md": VIAGEM_BUENOS_AIRES,
    "notas_aula_fisica.md": NOTAS_FISICA,
}


def write_pdf(path: Path) -> None:
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(56, 56, 540, 790)
    y_used = 0.0
    for paragraph in CONTRATO_PARAGRAFOS:
        # insert_textbox devolve o espaço que sobrou (negativo se não coube)
        left = -1.0
        if rect.y0 + y_used < rect.y1 - 30:
            box = fitz.Rect(rect.x0, rect.y0 + y_used, rect.x1, rect.y1)
            left = page.insert_textbox(box, paragraph, fontsize=11, fontname="helv")
        if left < 0:
            page = doc.new_page()
            y_used = 0.0
            box = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1)
            left = page.insert_textbox(box, paragraph, fontsize=11, fontname="helv")
        y_used = (rect.y1 - rect.y0) - left + 14
    doc.set_metadata({"title": "Contrato de locação (fictício)", "author": "Jarvis evals"})
    doc.save(str(path))
    doc.close()


def write_docx(path: Path) -> None:
    from docx import Document

    document = Document()
    document.add_heading(RELATORIO_PARAGRAFOS[0], level=1)
    for paragraph in RELATORIO_PARAGRAFOS[1:]:
        document.add_paragraph(paragraph)
    table = document.add_table(rows=len(RELATORIO_TABELA), cols=3)
    for row_index, row in enumerate(RELATORIO_TABELA):
        for col_index, value in enumerate(row):
            table.cell(row_index, col_index).text = value
    document.core_properties.author = "Jarvis evals"
    document.save(str(path))


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in TEXT_FILES.items():
        with open(str(FIXTURES_DIR / name), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    write_pdf(FIXTURES_DIR / "contrato_aluguel.pdf")
    write_docx(FIXTURES_DIR / "relatorio_trimestral.docx")
    for path in sorted(FIXTURES_DIR.iterdir()):
        print("{:32s} {:>7d} bytes".format(path.name, path.stat().st_size))
    return 0


if __name__ == "__main__":
    sys.exit(main())

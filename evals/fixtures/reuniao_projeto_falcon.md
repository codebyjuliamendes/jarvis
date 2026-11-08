# Reunião do projeto Falcon — 14/10/2026

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

## Detalhes técnicos discutidos

Rafael explicou que a migração troca o SQLite por PostgreSQL 16 e cria um índice único por sala e horário. O teste de carga da semana passada derrubou o servidor com 320 reservas por minuto; a meta é aguentar 500 sem erro e com tempo de resposta abaixo de 300 milissegundos.

Bianca mostrou o protótipo do novo cancelamento: a pessoa cancela com dois toques e a sala volta a aparecer como livre na hora. Também entra um lembrete por e-mail 15 minutos antes da reunião, que pode ser desligado nas preferências.

## Métricas de uso

Em setembro foram 4.870 reservas, 18% delas canceladas. A sala mais disputada é a Aurora, com taxa de ocupação de 92% entre 10h e 16h. Cerca de 37% das reservas são feitas pelo celular.

## Pendências da reunião anterior

A integração com o crachá para liberar a porta da sala continua parada esperando o time de segurança. O relatório mensal de ocupação para a diretoria foi entregue no dia 2 de outubro.

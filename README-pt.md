# laya-mcp

[Laya](https://github.com/NandhaKishorM/laya) é um modelo de decisão «System 1» rápido e não autorregressivo: responde a perguntas tipadas — `noul` (sim/não), `choice`, `score` — sobre um estado e devolve probabilidades numa única passagem forward. É genuinamente bom, e é um artefacto de investigação.

Isto é a parte que o faz sobreviver ao contacto com um servidor.

[English](README.md) · [简体中文](README-zh.md) · [Español](README-es.md) · [Português](README-pt.md) · [हिन्दी](README-hi.md)

```bash
pip install 'laya-mcp[mcp]'
laya-mcp serve            # carrega o modelo uma vez, mantém-no quente em 127.0.0.1:8787
laya-mcp install          # regista-o no harness de agente que tiveres
```

> **Estado: 0.2.2, em desenvolvimento.** O núcleo está implementado e a sua lógica pura está coberta por 94 verificações, mas ainda não foi exercitado de ponta a ponta contra um harness real em CI. As interfaces podem mudar antes de 1.0.

---

## O problema que resolve

O Laya trunca coisas em silêncio, e as omissões são daquele tipo que só notas depois de agir sobre uma resposta errada.

**Corta o estado, pelo fim, e não diz nada.** `build_sequence` dá ao estado o espaço que sobra depois das opções e corta-o com `st[:room]`. Um documento longo perde a cauda — que num contrato, num log ou numa thread de email é muitas vezes onde estava a resposta — e o modelo responde então sobre o prefixo sobrevivente com total confiança. Nada na resposta o assinala.

**Encurta as opções até as etiquetas serem indistinguíveis.** As opções partilham um orçamento fixo de `head_max_len` (192 tokens no checkpoint inglês, 256 nos restantes). A partir de certo ponto cada etiqueta recebe ~4 tokens. É a causa documentada do colapso do Banking77 (0.425 contra 0.870 do Jev), e mais uma vez nada o reporta.

**A sua validação é uma única verificação.** Um `type` desconhecido é um `KeyError` cru de `QTYPES[q["t"]]`; um `criteria` em falta também é um `KeyError` cru. São indistinguíveis de um bug do modelo, e nenhum nomeia a pergunta culpada.

**O seu `confidence` não é exatidão.** É `1 - H(p)/log(k)` para `choice` e `score`, e `max(p, 1-p)` para `noul`. Uma entropia normalizada é *baixa* quando a probabilidade está espalhada mesmo quando a opção principal está certa, e *alta* numa resposta errada mas confiante — o checkpoint inglês obtém 0.000 de exatidão em khmer com 0.952 de confiança. Um limiar sobre ela não significa o que parece.

**E degrada-se para CPU em silêncio.** Perante um OOM de CUDA move o modelo para CPU em fp32, no local, permanentemente, imprimindo para stdout. Não fica nenhuma flag em lado nenhum. Um processo que apanha isto uma vez continua a responder, cerca de 10–15x mais lento, e nada na resposta o admite.

Por isso este pacote acrescenta o que falta: um **preflight** que diz o que seria cortado, **erros estruturados** que nomeiam a pergunta, um **contrato de confiança honesto**, e uma **superfície de saúde** que reporta uma degradação.

---

## O que faz

| | |
|---|---|
| **Sidecar quente** | Um `Router`, pré-carregado, durante toda a vida do processo. O padrão do Laya (`max_loaded=1`) reconstrói um modelo a cada mudança de idioma — medido a montante em 7.4 s de recarga mediana em CPU, 10.3 s numa T4. |
| **Preflight de orçamento de tokens** | O `laya_plan` reporta exatamente o que seria truncado e quantos tokens cada opção recebe de facto, **sem correr o modelo**. A aritmética das opções reproduz o `build_sequence` linha por linha. |
| **Erros estruturados** | Cada falha do Laya torna-se um código, um estado HTTP, o id da pergunta culpada e uma dica. `KeyError('ranking')` torna-se `invalid_question` a nomear o tipo. |
| **Confiança honesta** | O `confidence` é rotulado pelo que é, em cada resposta. As respostas `noul` trazem também uma banda `no` / `uncertain` / `yes`, porque uma probabilidade calibrada não é uma decisão. |
| **Armazém de calibração** | Ajusta uma temperatura por `(primitive, bucket de opções)` contra as tuas próprias etiquetas, persiste-a, recarrega-a. O `laya-multilingual` **não** traz nenhuma temperatura ajustada, por isso as suas probabilidades são cruas até o fazeres. |
| **Honestidade do dispositivo** | Reporta uma degradação silenciosa para CPU, e o `doctor` prova que a GPU funciona correndo uma operação real em vez de confiar em `torch.cuda.is_available()`. |
| **Inferência serializada** | Um lock, por defeito. O Laya não é thread-safe: o `system_one` reatribui `self.device` e chama `self.model.to(...)` perante um OOM, por isso chamadas concorrentes podem competir com uma mudança de dispositivo. |
| **Reiniciável** | `DELETE /model` liberta o modelo e esvazia a cache do alocador CUDA, que o `Router.unload` não esvazia. Um servidor de modelos que fuga precisa de ser reciclável. |
| **Um `noul` que não é uma constante** | O Laya renderiza cada `noul` como `false: ...` / `true: ...` e depois responde «false» a praticamente todos — 40 de 40 itens, em ambos os idiomas, exatamente ao nível do acaso. O que se rompe é a palavra da etiqueta, não a primitiva, por isso um `noul` que traz um boundary é perguntado como uma escolha de duas opções com etiquetas neutras e lido de volta como `P(true)`: **0.500 → 1.000** (inglês) e **0.975** (multilingue) sobre esses mesmos quarenta itens. Um `noul` sem boundary é enviado sem alterações e a resposta diz porquê. |

---

## Um instalador, seis harnesses

Não há forma portátil de registar um servidor MCP. Medidos contra harnesses reais instalados, discordam no ficheiro, no formato e na chave:

| harness | configuração | formato | chave |
|---|---|---|---|
| Claude Code | `~/.claude.json` | JSON | `mcpServers` |
| Cursor | `~/.cursor/mcp.json` | JSON | `mcpServers` |
| Codex | `~/.codex/config.toml` | TOML | `[mcp_servers.<name>]` |
| opencode | `~/.config/opencode/opencode.json[c]` | JSON | `mcp` |
| OpenClaw | `~/.openclaw/openclaw.json` | JSON | `mcp.servers` |
| Hermes | `HERMES_HOME`, senão `%LOCALAPPDATA%\hermes` no Windows ou `~/.hermes` | YAML | `mcp_servers` |

O `laya-mcp install` deteta quais estão presentes e escreve a forma correta em cada um. Todos os escritores fundem em vez de substituir, fazem cópia de segurança primeiro, e recusam tocar num ficheiro que não conseguem analisar — o `~/.claude.json` é um ficheiro partilhado grande que guarda histórico e estado por projeto, e sobrescrevê-lo para instalar um modelo de decisão seria uma troca catastrófica.

Duas limitações honestas:

* **O opencode difere de todos mais três vezes** dentro da sua própria entrada: `command` é um único array que contém o executável *e* os seus argumentos, a chave de ambiente é `environment`, não `env`, e o interruptor é `enabled`. Pôr `disabled: true` lá é ignorado em silêncio.
* **O `pi` não é suportado.** Não é um descuido: o `pi` não tem suporte MCP nativo. A sua referência de definições não contém nenhuma chave MCP, e o seu próprio pedido a montante sobre MCP intitula-se *«Add MCP extension example»* — no `pi`, MCP é uma extensão que construis. Não há ficheiro de configuração que um instalador possa escrever. O `install` deteta-o e di-lo.

O `install` aponta o harness para `python -m laya_mcp mcp` em vez do script de consola `laya-mcp`, deliberadamente: no Windows um script de consola é um shim `.cmd` e o SDK de MCP lança processos com `shell: false`, que não o consegue executar.

### Skills

Registar o servidor é só metade da instalação. Sem a skill, o harness vê cinco ferramentas com descrições de um parágrafo e nenhuma das regras que decidem se uma resposta significa alguma coisa:

```bash
laya-mcp install --with-skill   # registo MCP + SKILL.md para cada harness encontrado
laya-mcp install --skill-only   # só o SKILL.md, sem registo do servidor
laya-mcp install --with-skill --harness cursor,claude  # só estes dois
laya-mcp install --skill-only --dry-run  # mostra os caminhos, não escreve nada
```

Todos os harnesses convergem em `<skills>/<nome>/SKILL.md`; só a raiz difere:

| harness | ficheiro da skill |
|---|---|
| Claude Code | `~/.claude/skills/laya/SKILL.md` |
| Cursor | `~/.cursor/skills/laya/SKILL.md` |
| Codex | `$CODEX_HOME/skills/laya/SKILL.md`, senão `~/.codex/skills/laya/SKILL.md` |
| opencode | `~/.config/opencode/skills/laya/SKILL.md` (`XDG_CONFIG_HOME` tem prioridade) |
| OpenClaw | `~/.openclaw/skills/laya/SKILL.md` |
| Hermes | `~/.hermes/skills/laya/SKILL.md` (`HERMES_HOME` senão o omisso da plataforma, como na configuração) |

Aplica-se o mesmo contrato de fusão/cópia de segurança do escritor de configuração, e voltar a correr é um no-op reportado como `unchanged` em vez de uma nova cópia de segurança. O texto instalado é o `SKILL.md` da raiz do repositório, incluído no wheel para que um `pip install` o consiga ler sem checkout; `--skill-source FICHEIRO` sobrepõe-no e `--skill-name NOME` renomeia a pasta (tem de coincidir com o `name` do frontmatter).

Com `--project DIR`, as skills de âmbito de projeto vão para `DIR/.agents/skills/laya/`, `DIR/.claude/skills/laya/` e `DIR/.cursor/skills/laya/` — uma escrita por forma nativa, porque `.agents/skills` é o diretório neutro que Cursor, Codex, opencode e OpenClaw lêem, enquanto Claude e Cursor preferem a sua própria raiz. O Hermes não tem âmbito de skill por projeto, por isso a instalação global acima é toda a história aí. O Cursor precisa de sair e reabrir por completo antes de uma skill nova aparecer; o Claude Code apanha-a em direto.

---

## Ferramentas

| ferramenta | o que responde |
|---|---|
| `laya_ask` | Um lote de perguntas tipadas sobre um estado. A geral. |
| `laya_noul` | Uma pergunta de sim/não. Devolve `P(true)` e uma banda. |
| `laya_choice` | Uma pergunta de escolha múltipla. Devolve a etiqueta e a distribuição. |
| `laya_score` | Uma pergunta de escala ordenada. |
| `laya_plan` | «Cabe, e o que será cortado?» — sem passagem forward. |

Cada descrição diz para que *não* serve a ferramenta. Um modelo de decisão a quem se pede para escrever prosa não produz nada de útil, e um agente que não saiba disso vai continuar a tentar.

---

## HTTP

```bash
laya-mcp serve --model english --port 8787
curl -s localhost:8787/health
curl -s localhost:8787/ask -H 'content-type: application/json' -d '{
  "state": {"subject": "Duplicate charge", "body": "Billed twice. Refund or we cancel."},
  "questions": {
    "churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
    "team":  {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": "invoices, refunds", "tech": "bugs, outages"}}
  }
}'
```

`GET /health`, `GET /capabilities`, `GET /version`, `POST /ask`, `POST /plan`, `DELETE /model`. Apenas loopback por defeito; ligar noutro sítio avisa em voz alta, porque não há autenticação.

O `POST /plan` aceita o mesmo corpo que o `/ask` e devolve o mesmo bloco `budget` que o `/ask` reporta, calculado pela mesma chamada a `plan_questions` — sem passagem forward. É assim que um cliente pode perguntar «isto vai ser cortado?» antes de pagar por uma resposta. Num host frio paga um *carregamento* do modelo, que não é o mesmo que uma inferência.

---

## Configuração que vale a pena conhecer

| opção | porquê |
|---|---|
| `--head-max-len` | Aumentada no arranque, é a solução para `choice` de alta cardinalidade. As opções partilham-na, por isso mais espaço por etiqueta é a única forma de as manter distinguíveis. É lida de novo em cada chamada, por isso defini-la uma vez chega. |
| `--max-len` | O orçamento total. Aumentá-lo é a solução para um estado truncado. |
| `--truncate-left` | Guarda o **fim** de um estado demasiado grande em vez do início. Desligada por defeito porque muda que parte de um documento longo o modelo lê — e decide respostas: um mesmo estado de 16 958 caracteres, com um engodo à frente e a correção atrás, deu um `noul` de **0.0706** guardando o início e **0.8341** guardando o fim. Usa-a quando a resposta estiver no fim (uma thread, um registo, as cláusulas finais de um contrato), e lê `truncated.state.kept` para saber que extremidade sobreviveu. |
| `--concurrency` | Aumenta só se souberes que o Laya não está a partilhar estado de dispositivo. O 1 por defeito é correção, não cautela. |
| `--sidecar` | Aponta o `laya-mcp mcp` a um `serve` em execução. Muito recomendado: um harness lança um servidor stdio por sessão, e alojar o modelo em cada um paga o custo de carregamento por sessão. |

---

## O que isto não corrige

Os números do próprio projeto a montante merecem ser repetidos, porque uma camada de integração que insinue o contrário está a mentir-te.

* **Os checkpoints base estão perto do acaso em zero-shot em decisões tipadas** — 0.362 para inglês contra uma **linha de base de classe maioritária de 0.461**. Adivinhar a resposta mais comum bate o modelo.
* **O `score` é a primitiva mais fraca.** Medida independentemente em 35% contra os 70% do Jev numa tarefa ordinal de cinco níveis.
* **A calibração precisa de dados etiquetados.** O ECE cru é 0.466 para inglês e 0.314 para multilingue, melhorando para 0.081 e 0.106 após o ajuste. Uma temperatura não se inventa; este pacote não finge o contrário. Um ajuste que embate no limite da sua grelha de busca é registado em `saturated_buckets` como um limite, e não devolvido como temperatura: um limite descreve a amostra, não o modelo.
* **O viés de posição é real.** Uma execução de fixture publicada respondeu «A» em 46 de 50 itens de escolha múltipla.
* **A exatidão cai acima de ~20 opções**, segundo o autor.

A calibração torna uma probabilidade *honesta*; não consegue tornar um modelo *correto*. Se a exatidão não está lá para a tua tarefa, ajusta com os teus próprios dados de domínio ou não o coloques em produção.

---

## Verificar

```bash
python tests/smoke_pure.py         # 94 verificações: validação, planeamento, calibração, erros
python tests/install_harnesses.py  # 38: cada dialeto de harness, num diretório temporário
python tests/mcp_protocol.py       # 37 com sidecar (32 sem ele): um handshake MCP real e chamadas de ferramenta reais
python tests/stdio_latency.py      # handshake <5 s, tools/list instantâneo, tools/call responde
python tests/language_probe.py     # o que cada checkpoint consegue realmente fazer, por idioma
laya-mcp doctor                    # o que está instalado, e o que a GPU consegue mesmo fazer
```

169 verificações nas três suites, e cada uma cobre uma camada que as outras não alcançam. O `stdio_latency.py` e o `language_probe.py` precisam de um modelo e são medições, não asserções, por isso correm-se à mão e os seus números são citados acima.

O `smoke_pure.py` não precisa de torch, modelo, rede ou configuração de harness. O `install_harnesses.py` redireciona cada harness para um diretório temporário, porque o `~/.claude.json` é um ficheiro partilhado grande que guarda histórico e estado por projeto, e um teste que o sobrescrevesse seria um bug pior do que qualquer um que pudesse apanhar.

O `mcp_protocol.py` é o que mais importa e o que esteve ausente durante mais tempo. Lança o servidor exatamente como um harness o faz (`python -m laya_mcp mcp`), realiza o handshake real de `initialize` com o SDK oficial, lista as ferramentas e chama-as. Dois defeitos reais escaparam às outras suites e só foram apanhados aqui: o `FastMCP` no mcp 1.30 não aceita um argumento `version`, por isso o servidor não arrancava de todo; e o aviso de orçamento de tokens estava escrito na descrição da ferramenta do plugin de DSH mas nunca na deste servidor, por isso um cliente a usar MCP não poderia saber que um estado demasiado grande é cortado pelo fim.

A aceitação foi depois verificada deixando os harnesses analisar **e ligar-se a** os ficheiros que esta ferramenta escreve, que é o único teste que distingue um ficheiro escrito de um ficheiro aceite:

| harness | como | resultado |
|---|---|---|
| opencode | `opencode mcp list` | ✓ connected |
| claude | `claude mcp list` | √ Connected |
| codex | `codex mcp list --json` | `enabled`, `"type": "stdio"`, argv correto — `auth_status: unsupported` não é uma falha: um servidor stdio local não precisa de autenticação |
| OpenClaw | `openclaw mcp list --json` | reporta o servidor e o transporte stdio |
| Hermes | `hermes mcp list` | ✓ enabled, e `hermes mcp test laya` liga e encontra as 5 ferramentas |
| `pi` | — | sem suporte MCP nativo; o `install` deteta-o e di-lo |

Todos os harnesses que este instalador suporta confirmam agora através das suas
próprias ferramentas. Essa é a única verificação que distingue um ficheiro escrito
de um ficheiro aceite, e ganhou o seu lugar: no Windows o Hermes lê a sua
configuração de `%LOCALAPPDATA%\hermes`, não de `~/.hermes`, por isso o instalador
reportava sucesso enquanto escrevia um ficheiro que ninguém lia. Dois defeitos
escondiam-no: o Hermes estava marcado como não verificável, e no Windows nenhum
destes listers podia sequer ser lançado, porque o npm entrega cada um como um shim
`.cmd` que o `CreateProcess` se recusa a executar.

Essa tabela é a razão de existir o `stdio_latency.py`. Todos os harnesses acima dão a um servidor MCP 30 segundos para terminar o `initialize`, e responder ao handshake só depois de carregar um checkpoint levava 19 s sem contenção e 275 s enquanto outro modelo ocupava a GPU — por isso todos reportavam «Failed to connect» sobre uma configuração que tinham analisado perfeitamente. Agora o carregamento corre por trás do handshake.

## Licença

Apache-2.0. O Laya é Apache-2.0, da Convai Innovations. Esta é uma integração independente e não está afiliada nem é endossada por esse projeto.

# laya-mcp

[Laya](https://github.com/NandhaKishorM/laya) es un modelo de decisión «System 1» rápido y no autorregresivo: responde preguntas tipadas — `noul` (sí/no), `choice`, `score` — sobre un estado y devuelve probabilidades en una sola pasada hacia adelante. Es genuinamente bueno, y es un artefacto de investigación.

Esto es la parte que lo hace sobrevivir al contacto con un servidor.

[English](README.md) · [简体中文](README-zh.md) · [Español](README-es.md) · [Português](README-pt.md) · [हिन्दी](README-hi.md)

```bash
pip install 'laya-mcp[mcp]'
laya-mcp serve            # carga el modelo una vez, lo mantiene caliente en 127.0.0.1:8787
laya-mcp install          # lo registra en el harness de agente que tengas
```

> **Estado: 0.2.2, en desarrollo.** El núcleo está implementado y su lógica pura está cubierta por 94 comprobaciones, pero todavía no se ha ejercitado de extremo a extremo contra un harness real en CI. Las interfaces pueden cambiar antes de 1.0.

---

## El problema que resuelve

Laya trunca cosas en silencio, y las omisiones son de ese tipo que solo notas después de actuar sobre una respuesta incorrecta.

**Corta el estado, por el final, y no dice nada.** `build_sequence` le da al estado el espacio que sobra tras las opciones y lo corta con `st[:room]`. Un documento largo pierde su cola — que en un contrato, un log o un hilo de correo suele ser donde estaba la respuesta — y el modelo responde entonces sobre el prefijo superviviente con total confianza. Nada en la respuesta lo señala.

**Acorta las opciones hasta que las etiquetas son indistinguibles.** Las opciones comparten un presupuesto fijo de `head_max_len` (192 tokens en el checkpoint inglés, 256 en los demás). Pasado cierto punto cada etiqueta recibe ~4 tokens. Es la causa documentada del colapso de Banking77 (0.425 frente a 0.870 de Jev), y de nuevo nada lo reporta.

**Su validación es una sola comprobación.** Un `type` desconocido es un `KeyError` pelado de `QTYPES[q["t"]]`; un `criteria` ausente también es un `KeyError` pelado. Son indistinguibles de un bug del modelo, y ninguno nombra la pregunta culpable.

**Su `confidence` no es la exactitud.** Es `1 - H(p)/log(k)` para `choice` y `score`, y `max(p, 1-p)` para `noul`. Una entropía normalizada es *baja* cuando la probabilidad está repartida aunque la opción principal sea la correcta, y *alta* en una respuesta incorrecta pero segura — el checkpoint inglés obtiene 0.000 de exactitud en jemer con 0.952 de confianza. Un umbral sobre ella no significa lo que parece.

**Y se degrada a CPU en silencio.** Ante un OOM de CUDA mueve el modelo a CPU en fp32, en el sitio, permanentemente, imprimiendo a stdout. No se activa ninguna bandera en ningún lado. Un proceso que sufre esto una vez sigue respondiendo, unas 10–15 veces más lento, y nada en la respuesta lo admite.

Así que este paquete añade lo que falta: un **preflight** que dice qué se cortaría, **errores estructurados** que nombran la pregunta, un **contrato de confianza honesto**, y una **superficie de salud** que reporta una degradación.

---

## Qué hace

| | |
|---|---|
| **Sidecar caliente** | Un `Router`, precargado, durante toda la vida del proceso. El valor por defecto de Laya (`max_loaded=1`) reconstruye un modelo en cada cambio de idioma — medido aguas arriba en 7.4 s de recarga mediana en CPU, 10.3 s en una T4. |
| **Preflight de presupuesto de tokens** | `laya_plan` informa exactamente qué se truncaría y cuántos tokens recibe realmente cada opción, **sin ejecutar el modelo**. La aritmética de opciones reproduce `build_sequence` línea por línea. |
| **Errores estructurados** | Cada fallo de Laya se convierte en un código, un estado HTTP, el id de la pregunta culpable y una pista. `KeyError('ranking')` se convierte en `invalid_question` nombrando el tipo. |
| **Confianza honesta** | `confidence` se etiqueta por lo que es, en cada respuesta. Las respuestas `noul` llevan además una banda `no` / `uncertain` / `yes`, porque una probabilidad calibrada no es una decisión. |
| **Almacén de calibración** | Ajusta una temperatura por `(primitive, bucket de opciones)` contra tus propias etiquetas, persístela, recárgala. `laya-multilingual` **no** trae ninguna temperatura ajustada, así que sus probabilidades son crudas hasta que lo hagas. |
| **Honestidad de dispositivo** | Reporta una degradación silenciosa a CPU, y `doctor` demuestra que la GPU funciona ejecutando una operación real en vez de confiar en `torch.cuda.is_available()`. |
| **Inferencia serializada** | Un lock, por defecto. Laya no es thread-safe: `system_one` reasigna `self.device` y llama a `self.model.to(...)` ante un OOM, así que las llamadas concurrentes pueden competir con un cambio de dispositivo. |
| **Reiniciable** | `DELETE /model` libera el modelo y vacía la caché del asignador de CUDA, que `Router.unload` no vacía. Un servidor de modelos que gotea necesita ser reciclable. |
| **Un `noul` que no es una constante** | Laya renderiza cada `noul` como `false: ...` / `true: ...` y luego responde «false» a prácticamente todos — 40 de 40 ítems, en ambos idiomas, exactamente el azar. Lo que se rompe es la palabra de la etiqueta, no la primitiva, así que un `noul` que lleva un boundary se pregunta como una elección de dos opciones con etiquetas neutras y se lee de vuelta como `P(true)`: **0.500 → 1.000** (inglés) y **0.975** (multilingüe) sobre esos mismos cuarenta ítems. Un `noul` sin boundary se envía sin cambios y la respuesta dice por qué. |

---

## Un instalador, seis harnesses

No hay forma portable de registrar un servidor MCP. Medidos contra harnesses reales instalados, discrepan en el archivo, el formato y la clave:

| harness | configuración | formato | clave |
|---|---|---|---|
| Claude Code | `~/.claude.json` | JSON | `mcpServers` |
| Cursor | `~/.cursor/mcp.json` | JSON | `mcpServers` |
| Codex | `~/.codex/config.toml` | TOML | `[mcp_servers.<name>]` |
| opencode | `~/.config/opencode/opencode.json[c]` | JSON | `mcp` |
| OpenClaw | `~/.openclaw/openclaw.json` | JSON | `mcp.servers` |
| Hermes | `HERMES_HOME`, si no `%LOCALAPPDATA%\hermes` en Windows o `~/.hermes` | YAML | `mcp_servers` |

`laya-mcp install` detecta cuáles están presentes y escribe la forma correcta en cada uno. Todos los escritores fusionan en vez de reemplazar, hacen copia de seguridad primero, y se niegan a tocar un archivo que no pueden parsear — `~/.claude.json` es un archivo compartido grande que guarda historial y estado por proyecto, y sobrescribirlo para instalar un modelo de decisión sería un trueque catastrófico.

Dos limitaciones honestas:

* **opencode difiere de todos tres veces más** dentro de su propia entrada: `command` es un único array que contiene el ejecutable *y* sus argumentos, la clave de entorno es `environment`, no `env`, y el interruptor es `enabled`. Poner `disabled: true` ahí se ignora en silencio.
* **`pi` no está soportado.** No es un descuido: `pi` no tiene soporte MCP nativo. Su referencia de ajustes no contiene ninguna clave MCP, y su propia petición aguas arriba sobre MCP se titula *«Add MCP extension example»* — en `pi`, MCP es una extensión que construyes tú. No hay archivo de configuración que un instalador pueda escribir. `install` lo detecta y lo dice.

`install` apunta el harness a `python -m laya_mcp mcp` en vez de al script de consola `laya-mcp`, deliberadamente: en Windows un script de consola es un shim `.cmd` y el SDK de MCP lanza procesos con `shell: false`, que no puede ejecutarlo.

### Skills

Registrar el servidor es solo la mitad de la instalación. Sin la skill, el harness ve cinco herramientas con descripciones de un párrafo y ninguna de las reglas que deciden si una respuesta significa algo:

```bash
laya-mcp install --with-skill   # registro MCP + SKILL.md para cada harness encontrado
laya-mcp install --skill-only   # solo el SKILL.md, sin registro del servidor
laya-mcp install --with-skill --harness cursor,claude  # solo estos dos
laya-mcp install --skill-only --dry-run  # muestra las rutas, no escribe nada
```

Todos los harnesses convergen en `<skills>/<nombre>/SKILL.md`; solo difiere la raíz:

| harness | archivo de la skill |
|---|---|
| Claude Code | `~/.claude/skills/laya/SKILL.md` |
| Cursor | `~/.cursor/skills/laya/SKILL.md` |
| Codex | `$CODEX_HOME/skills/laya/SKILL.md`, si no `~/.codex/skills/laya/SKILL.md` |
| opencode | `~/.config/opencode/skills/laya/SKILL.md` (`XDG_CONFIG_HOME` tiene prioridad) |
| OpenClaw | `~/.openclaw/skills/laya/SKILL.md` |
| Hermes | `~/.hermes/skills/laya/SKILL.md` (`HERMES_HOME` si no el valor de la plataforma, como en la configuración) |

Se aplica el mismo contrato de fusión/copia de seguridad que el escritor de configuración, y volver a ejecutarlo es un no-op reportado como `unchanged` en vez de una copia nueva cada vez. El texto instalado es el `SKILL.md` de la raíz del repositorio, incluido en el wheel para que un `pip install` pueda leerlo sin checkout; `--skill-source ARCHIVO` lo sustituye y `--skill-name NOMBRE` renombra la carpeta (debe coincidir con el `name` del frontmatter).

Con `--project DIR`, las skills de ámbito de proyecto van a `DIR/.agents/skills/laya/`, `DIR/.claude/skills/laya/` y `DIR/.cursor/skills/laya/` — una escritura por forma nativa, porque `.agents/skills` es el directorio neutro que Cursor, Codex, opencode y OpenClaw leen, mientras Claude y Cursor prefieren su propia raíz. Hermes no tiene ámbito de skill por proyecto, así que la instalación global de arriba es toda la historia ahí. Cursor necesita salir y reabrir por completo antes de que aparezca una skill nueva; Claude Code la recoge en directo.

---

## Herramientas

| herramienta | qué responde |
|---|---|
| `laya_ask` | Un lote de preguntas tipadas sobre un estado. La general. |
| `laya_noul` | Una pregunta de sí/no. Devuelve `P(true)` y una banda. |
| `laya_choice` | Una pregunta de opción múltiple. Devuelve la etiqueta y la distribución. |
| `laya_score` | Una pregunta de escala ordenada. |
| `laya_plan` | «¿Cabe, y qué se cortará?» — sin pasada hacia adelante. |

Cada descripción dice para qué *no* sirve la herramienta. Un modelo de decisión al que se le pide escribir prosa no produce nada útil, y un agente que no lo sepa seguirá intentándolo.

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

`GET /health`, `GET /capabilities`, `GET /version`, `POST /ask`, `POST /plan`, `DELETE /model`. Solo loopback por defecto; enlazar en otro sitio avisa a gritos, porque no hay autenticación.

`POST /plan` acepta el mismo cuerpo que `/ask` y devuelve el mismo bloque `budget` que reporta `/ask`, calculado por la misma llamada a `plan_questions` — sin pasada hacia adelante. Es como un cliente puede preguntar «¿se cortará esto?» antes de pagar por una respuesta. En un host frío paga una *carga* de modelo, que no es lo mismo que una inferencia.

---

## Configuración que conviene conocer

| opción | por qué |
|---|---|
| `--head-max-len` | Elevada al arrancar, es la solución para `choice` de alta cardinalidad. Las opciones la comparten, así que más espacio por etiqueta es la única forma de mantenerlas distinguibles. Se lee de nuevo en cada llamada, así que fijarla una vez basta. |
| `--max-len` | El presupuesto total. Elevarla es la solución para un estado truncado. |
| `--truncate-left` | Conserva el **final** de un estado sobredimensionado en lugar de su principio. Desactivada por defecto porque cambia qué parte de un documento largo lee el modelo — y decide respuestas: un mismo estado de 16 958 caracteres, con un señuelo delante y la corrección detrás, dio un `noul` de **0.0706** conservando el principio y **0.8341** conservando el final. Úsala cuando la respuesta esté al final (un hilo, un registro, las cláusulas finales de un contrato), y lee `truncated.state.kept` para saber qué extremo sobrevivió. |
| `--concurrency` | Súbela solo si sabes que Laya no comparte estado de dispositivo. El 1 por defecto es corrección, no cautela. |
| `--sidecar` | Apunta `laya-mcp mcp` a un `serve` en ejecución. Muy recomendado: un harness lanza un servidor stdio por sesión, y alojar el modelo en cada uno paga el coste de carga por sesión. |

---

## Lo que esto no arregla

Los números del propio proyecto aguas arriba merecen repetirse, porque una capa de integración que insinúe lo contrario te está mintiendo.

* **Los checkpoints base están cerca del azar en zero-shot sobre decisiones tipadas** — 0.362 para inglés frente a una **línea base de clase mayoritaria de 0.461**. Adivinar la respuesta más común supera al modelo.
* **`score` es la primitiva más débil.** Medida independientemente en 35% frente al 70% de Jev en una tarea ordinal de cinco niveles.
* **La calibración necesita datos etiquetados.** El ECE crudo es 0.466 para inglés y 0.314 para multilingüe, mejorando a 0.081 y 0.106 tras el ajuste. Una temperatura no se puede inventar; este paquete no pretende lo contrario. Un ajuste que choca con el borde de su rejilla de búsqueda se registra en `saturated_buckets` como una cota, no se devuelve como temperatura: una cota describe la muestra, no el modelo.
* **El sesgo de posición es real.** Una ejecución de fixture publicada respondió «A» en 46 de 50 ítems de opción múltiple.
* **La exactitud decae por encima de ~20 opciones**, según el autor.

La calibración hace que una probabilidad sea *honesta*; no puede hacer que un modelo sea *correcto*. Si la exactitud no está ahí para tu tarea, ajusta con tus propios datos de dominio o no lo despliegues.

---

## Verificar

```bash
python tests/smoke_pure.py         # 94 comprobaciones: validación, planificación, calibración, errores
python tests/install_harnesses.py  # 38: cada dialecto de harness, en un directorio temporal
python tests/mcp_protocol.py       # 37 con sidecar (32 sin él): un handshake MCP real y llamadas de herramienta reales
python tests/stdio_latency.py      # handshake <5 s, tools/list instantáneo, tools/call responde
python tests/language_probe.py     # qué puede hacer realmente cada checkpoint, por idioma
laya-mcp doctor                    # qué está instalado, y qué puede hacer de verdad la GPU
```

169 comprobaciones en las tres suites, y cada una cubre una capa que las otras no alcanzan. `stdio_latency.py` y `language_probe.py` necesitan un modelo y son mediciones, no aserciones, así que se ejecutan a mano y sus números se citan arriba.

`smoke_pure.py` no necesita torch, modelo, red ni configuración de harness. `install_harnesses.py` redirige cada harness a un directorio temporal, porque `~/.claude.json` es un archivo compartido grande que guarda historial y estado por proyecto, y un test que lo sobrescribiera sería un bug peor que cualquiera que pudiera detectar.

`mcp_protocol.py` es el que más importa y el que estuvo ausente más tiempo. Lanza el servidor exactamente como lo hace un harness (`python -m laya_mcp mcp`), realiza el handshake real de `initialize` con el SDK oficial, lista las herramientas y las llama. Dos defectos reales escaparon a las otras suites y solo se atraparon aquí: `FastMCP` en mcp 1.30 no acepta un argumento `version`, así que el servidor no arrancaba en absoluto; y el aviso de presupuesto de tokens estaba escrito en la descripción de la herramienta del plugin de DSH pero nunca en la de este servidor, así que un cliente que usara MCP no habría podido saber que un estado demasiado grande se corta por el final.

La aceptación se verificó después dejando que los harnesses parsearan **y se conectaran a** los archivos que escribe esta herramienta, que es la única prueba que distingue un archivo escrito de un archivo aceptado:

| harness | cómo | resultado |
|---|---|---|
| opencode | `opencode mcp list` | ✓ connected |
| claude | `claude mcp list` | √ Connected |
| codex | `codex mcp list --json` | `enabled`, `"type": "stdio"`, argv correcto — `auth_status: unsupported` no es un fallo: un servidor stdio local no necesita autenticación |
| OpenClaw | `openclaw mcp list --json` | reporta el servidor y el transporte stdio |
| Hermes | `hermes mcp list` | ✓ enabled, y `hermes mcp test laya` conecta y encuentra las 5 herramientas |
| `pi` | — | sin soporte MCP nativo; `install` lo detecta y lo dice |

Todos los harnesses que este instalador soporta confirman ahora a través de sus
propias herramientas. Esa es la única comprobación que distingue un archivo escrito
de uno aceptado, y se ha ganado su sitio: en Windows Hermes lee su configuración de
`%LOCALAPPDATA%\hermes`, no de `~/.hermes`, así que el instalador informaba de éxito
mientras escribía un archivo que nadie leía. Dos fallos lo ocultaban: Hermes estaba
marcado como no verificable, y en Windows no se podía ni lanzar estos listers,
porque npm entrega cada uno como un shim `.cmd` que `CreateProcess` se niega a
ejecutar.

Esa tabla es la razón de que exista `stdio_latency.py`. Todos los harnesses de arriba dan a un servidor MCP 30 segundos para terminar `initialize`, y responder al handshake solo después de cargar un checkpoint tardaba 19 s sin contención y 275 s mientras otro modelo ocupaba la GPU — así que todos reportaban «Failed to connect» sobre una configuración que habían parseado perfectamente. Ahora la carga corre por detrás del handshake.

## Licencia

Apache-2.0. Laya es Apache-2.0, de Convai Innovations. Esta es una integración independiente y no está afiliada ni respaldada por ese proyecto.

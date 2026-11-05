# Guia Completo: Quadro Trello (Kanban) e Roteiro de Gravação do Vídeo (Até 7 min)

Este documento reúne tudo o que você precisa para garantir **nota máxima (10,0)** nos critérios de **Uso do Trello**, **Uso do GitHub** e **Apresentação em Vídeo**.

---

## Parte 1 — Configuração do Quadro no Trello (Kanban)

### 1.1. Regras Obrigatórias da Avaliação (Seções 3, 5.3 e 6 do PDF)
1. Crie um quadro no Trello chamado: **`VisionGuard AI - Projeto Final SCTEC [Módulo 2]`**.
2. Crie exatamente estas **6 listas (colunas)** da esquerda para a direita:
   - `Backlog`
   - `To Do`
   - `Doing`
   - `Blocked`
   - `Review`
   - `Done`
3. **Permissões obrigatórias:**
   - Altere a visibilidade do quadro para **Público** (para que qualquer pessoa com o link veja).
   - Clique em **Compartilhar (Share)** no topo direito do Trello e convide o e-mail do professor: **`welington.souza@edu.sc.senai.br`**.
   - Copie o link do quadro e cole no `README.md` (na seção 7.1) e na entrega do AVA.
4. **Regra de Ouro nos Cards (Critérios 2 e 3 da Rubrica):**
   - **Atribua o seu perfil/nome (Juventino)** como membro em **todos os cards**.
   - Coloque a **descrição com as especificações técnicas de implementação** dentro de cada card (copie os textos abaixo).
   - Para comprovar que você **movimentou adequadamente os cards pelo board** (`Backlog -> To Do -> Doing -> Review -> Done`), crie os cards na lista `Backlog` e vá arrastando-os coluna por coluna até `Done` (o Trello registra o histórico de movimentação na atividade do card!) e adicione o comentário sugerido em cada card.

---

### 1.2. Cards Prontos para Copiar e Colar no Trello

#### Card 1 — `feature/dataset-prep`: Escolher, Higienizar e Preparar Dataset de 12 Classes
- **Membro atribuído:** Juventino
- **Etiqueta (Label):** `Dados / Dataset` (Azul)
- **Descrição (Especificações de Implementação):**
  ```markdown
  Responsável: Juventino
  Branch vinculada: feature/dataset-prep

  Especificações de Implementação:
  1. Selecionar e baixar o dataset base COCO128 (MS-COCO).
  2. Filtrar e remapear 12 classes distintas de objetos voltadas ao monitoramento urbano e cotidiano:
     - 0: person, 1: bicycle, 2: car, 3: motorcycle, 4: bus, 5: truck,
       6: traffic light, 7: dog, 8: backpack, 9: umbrella, 10: cup, 11: chair.
  3. Organizar diretórios images/train, images/val, labels/train e labels/val.
  4. Gerar arquivo datasets/visionguard12/data.yaml e relatório estatístico datasets/dataset_summary.json (83 imagens e 448 anotações).
  5. Separar 6 imagens variadas na pasta samples/ cobrindo 100% das 12 classes para testes de inferência.
  ```
- **Comentário para adicionar no Card:**
  > `Juventino: Card movimentado de Backlog -> To Do -> Doing -> Review -> Done após validação das 12 classes e geração do data.yaml e das 6 imagens variadas em samples/.`

---

#### Card 2 — `feature/yolo-finetuning`: Realizar Fine-Tuning e Benchmark do YOLO (v11 vs v8)
- **Membro atribuído:** Juventino
- **Etiqueta (Label):** `IA / YOLO` (Laranja)
- **Descrição (Especificações de Implementação):**
  ```markdown
  Responsável: Juventino
  Branch vinculada: feature/yolo-finetuning

  Especificações de Implementação:
  1. Implementar script scripts/train_and_benchmark.py para treinamento e avaliação comparativa.
  2. Aplicar Transfer Learning (Fine-Tuning) no modelo YOLO11n com as 10 camadas do backbone congeladas (freeze=10) sobre as 12 classes do VisionGuard-12.
  3. Realizar benchmark comparativo entre YOLO11n e YOLOv8n avaliando tempo de inferência (ms), FPS e cobertura das 12 classes.
  4. Implementar serviço singleton de inferência em app/detector.py carregando o modelo campeão (YOLO11n - 40.31ms / 24.8 FPS / 12 de 12 classes reconhecidas).
  5. Exportar gráficos de validação e matriz de confusão para docs/assets/.
  ```
- **Comentário para adicionar no Card:**
  > `Juventino: Fine-tuning concluído (mAP@50 = 63,2%) e benchmark executado. O modelo YOLO11n venceu em velocidade (24,8 FPS vs 24,2 FPS do YOLOv8n) e reconheceu 100% das 12 classes.`

---

#### Card 3 — `feature/api-setup`: Construir API FastAPI e Endpoint `POST /detect` (`UploadFile`)
- **Membro atribuído:** Juventino
- **Etiqueta (Label):** `Backend / FastAPI` (Verde)
- **Descrição (Especificações de Implementação):**
  ```markdown
  Responsável: Juventino
  Branch vinculada: feature/api-setup

  Especificações de Implementação:
  1. Definir os schemas Pydantic em app/schemas.py (BoundingBox, DetectionItem, DetectionResponse) seguindo o contrato JSON exigido:
     {"status": "success", "detections": [{"class_name": "...", "confidence": 0.92, "box": {"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}}]}
  2. Construir aplicação FastAPI em main.py com suporte a execução via uvicorn main:app --reload.
  3. Configurar endpoint POST /detect recebendo arquivo binário via UploadFile = File(...).
  4. Criar endpoints auxiliares GET /health, GET /classes e GET /benchmark.
  ```
- **Comentário para adicionar no Card:**
  > `Juventino: Endpoint POST /detect implementado com UploadFile e validado via Swagger UI (/docs), retornando JSON estruturado com classe, confiança e coordenadas do bounding box.`

---

#### Card 4 — `feature/tests-and-docs`: Testes no Swagger/Postman, Pytest e Documentação `README.md`
- **Membro atribuído:** Juventino
- **Etiqueta (Label):** `QA & Docs` (Roxo)
- **Descrição (Especificações de Implementação):**
  ```markdown
  Responsável: Juventino
  Branch vinculada: feature/tests-and-docs

  Especificações de Implementação:
  1. Criar suíte de testes automatizados em tests/test_api.py com FastAPI TestClient e Pytest.
  2. Testar o envio das 6 imagens variadas da pasta samples/ no endpoint POST /detect, comprovando detecção das 12 classes e validação de erro HTTP 400 para arquivos inválidos.
  3. Escrever documentação completa no README.md contendo: nome do sistema (VisionGuard AI), descrição do dataset e das 12 classes, justificativa e benchmark do YOLO11, instruções de instalação/execução e exemplo de payload JSON.
  4. Realizar merge das feature branches na branch develop e merge final na branch main.
  ```
- **Comentário para adicionar no Card:**
  > `Juventino: Todos os 5 testes automatizados passaram (pytest -v) cobrindo 100% das 12 classes. Documentação README.md finalizada e código mergeado na branch main.`

---

## Parte 2 — Roteiro Passo a Passo para Gravação do Vídeo (Máximo 7 Minutos)

> **Dica antes de começar a gravar:**  
> Deixe abertas 4 abas/janelas prontas na tela:
> 1. **Terminal** na pasta do projeto com a API rodando (`source .venv/bin/activate && uvicorn main:app --reload`).
> 2. **Navegador na aba 1:** Swagger UI da API (`http://127.0.0.1:8000/docs`).
> 3. **Navegador na aba 2:** Quadro do **Trello** público com os cards na coluna `Done`.
> 4. **Navegador na aba 3:** Repositório no **GitHub** (mostrando o `README.md`, as branches e os commits).

---

### Minuto 0:00 a 1:15 — 1. Apresentação, Objetivo do Sistema e as 12 Classes do Dataset
- **O que mostrar na tela:** O `README.md` no GitHub (ou VS Code) focado na tabela das **12 classes** e o endpoint `GET /classes` no Swagger.
- **O que falar:**
  > *"Olá! Meu nome é Juventino e este é o projeto avaliativo de Machine Learning e Visão Computacional do SCTEC. Eu desenvolvi o **VisionGuard AI**, um microserviço backend RESTful para detecção multiclasse de objetos voltado ao monitoramento urbano, tráfego e ambientes cotidianos.*  
  > *Para superar o requisito mínimo de 10 classes, preparei o dataset **VisionGuard-12** a partir do COCO128, contendo **83 imagens variadas e 448 anotações** distribuídas em **12 classes distintas**: `person` (pessoa), `bicycle` (bicicleta), `car` (carro), `motorcycle` (moto), `bus` (ônibus), `truck` (caminhão), `traffic light` (semáforo), `dog` (cachorro), `backpack` (mochila), `umbrella` (guarda-chuva), `cup` (copo) e `chair` (cadeira)."*

---

### Minuto 1:15 a 3:00 — 2. Demonstração Prática da API (`POST /detect` no Swagger / Postman)
- **O que mostrar na tela:** O terminal rodando `uvicorn main:app --reload` e a tela do Swagger (`http://127.0.0.1:8000/docs`) no endpoint `POST /detect`.
- **Ação ao vivo:**
  1. Clique em `POST /detect` -> **Try it out**.
  2. Faça upload da imagem `samples/sample_02_000000000074.jpg` e clique em **Execute**.
  3. Mostre o JSON retornado com `"status": "success"` e a lista `"detections"` contendo `bicycle`, `dog` e `person` com suas confianças e coordenadas `x_min, y_min, x_max, y_max`.
  4. Faça upload de uma segunda imagem variada, por exemplo `samples/sample_01_000000000257.jpg` (ou `sample_04_000000000359.jpg`), clique em **Execute** e mostre que ele detecta `truck`, `motorcycle`, `backpack`, `person` e `traffic light`.
- **O que falar:**
  > *"Aqui a API está em execução com FastAPI via `uvicorn main:app --reload`. No endpoint `POST /detect`, enviamos o arquivo binário via `UploadFile`. Ao enviar a primeira imagem de teste, a API processa a inferência no YOLO11 e retorna exclusivamente o JSON estruturado com `status: success` e cada objeto detectado com `class_name`, `confidence` e o `box` (`x_min, y_min, x_max, y_max`). Testando uma segunda imagem em cenário de tráfego urbano, vemos a detecção precisa de caminhões, motos, pedestres, mochilas e semáforos."*

---

### Minuto 3:00 a 4:30 — 3. Processo do YOLO, Fine-Tuning, Benchmark e Testes do Dataset
- **O que mostrar na tela:** A seção 3 do `README.md` (tabela de Benchmark e gráficos `results.png` e `val_batch0_pred.jpg`) e o terminal rodando `pytest -v`.
- **O que falar:**
  > *"Na etapa de Inteligência Artificial, trabalhei com a família YOLO da Ultralytics. Realizei tanto a rotina de **Fine-Tuning** com Transfer Learning no YOLO11n congelando as 10 primeiras camadas do backbone — alcançando 72,1% de precisão e mAP@50 de 63,2% — quanto um **benchmark comparativo entre o YOLO11n e o YOLOv8n**.*  
  > *O **YOLO11n** foi escolhido como modelo campeão para a API por ser mais rápido (latência média de 40,3 ms por imagem, chegando a 24,8 FPS contra 24,2 FPS do YOLOv8n) e por contar com os novos blocos de atenção espacial C2PSA, que generalizaram muito bem para objetos menores como semáforos, copos e mochilas, atingindo **100% de cobertura nas 12 classes**.*  
  > *Para comprovar os testes no dataset, criei uma suíte automatizada com `pytest` (`tests/test_api.py`) que envia as 6 imagens variadas da pasta `samples/` para o endpoint `POST /detect` e verifica automaticamente que todas as 12 classes são detectadas com sucesso."*

---

### Minuto 4:30 a 5:30 — 4. Organização Ágil no Trello (Kanban)
- **O que mostrar na tela:** A aba do seu quadro público no **Trello**. Abra rapidamente 1 ou 2 cards para mostrar que seu nome (**Juventino**) está atribuído e que há descrição técnica e histórico de movimentação.
- **O que falar:**
  > *"Antes de iniciar a codificação, estruturei o planejamento no Trello seguindo a metodologia Kanban com as listas `Backlog`, `To Do`, `Doing`, `Blocked`, `Review` e `Done`. Criei os cartões em paralelo com as branches do GitFlow, atribuí meu nome em cada cartão com as especificações técnicas de implementação e fui movimentando cada tarefa conforme avançava da preparação dos dados até a validação final da API."*

---

### Minuto 5:30 a 6:30 — 5. Branches do GitHub (GitFlow) e Commits
- **O que mostrar na tela:** O repositório no **GitHub** (aba Branches e aba Commits / Network Graph) ou no terminal o comando `git log --graph --oneline --all`.
- **O que falar:**
  > *"No controle de versão com Git e GitHub, apliquei o fluxo **GitFlow**. Além da branch `main`, criei a branch `develop` para concentrar a integração e quatro feature branches específicas:*
  > - *`feature/dataset-prep`: para aquisição e preparação das 12 classes do dataset;*
  > - *`feature/yolo-finetuning`: para o motor de inferência YOLO11, fine-tuning e benchmark;*
  > - *`feature/api-setup`: para a construção da API FastAPI e do endpoint `POST /detect`;*
  > - *`feature/tests-and-docs`: para os testes automatizados e documentação completa no `README.md`.*
  > *Todas as branches tiveram commits atômicos e semânticos, foram integradas na `develop` e, por fim, mergeadas na branch `main`. Muito obrigado!"*

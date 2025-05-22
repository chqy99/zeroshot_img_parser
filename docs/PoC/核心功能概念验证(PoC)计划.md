# 核心功能概念验证 (PoC) 计划

## 1. 引言

本文档旨在为桌面端智能代理系统的核心功能构建一份概念验证（Proof of Concept, PoC）计划。本系统的核心目标是基于零样本图像识别和向量检索能力实现对桌面屏幕内容的智能感知与理解，并在此基础上构建可检索的图像知识库，从而赋能未来的桌面自动化和智能辅助应用。

进行 PoC 的目的在于：

*   验证关键技术可行性：特别验证 SAM2, GroundingDINO, BLIP/CLIP 等零样本模型协同处理桌面 UI 截图的 Mixer 效果，以及向量数据库在存储 `ImageAnalysisResult` 并进行高效检索的可行性。
*   评估核心流程性能：初步评估从截图捕获到生成结构化分析结果（零样本路径）以及从截图捕获到检索到已知结果（检索优先路径）的关键环节的响应速度。
*   识别潜在技术难题：在实际环境中识别模型推理、模块集成、数据结构设计、索引性能等方面的具体问题。
*   指导后续详细设计与开发：通过 PoC 的成果和经验，为系统的详细设计和全面开发提供坚实的技术基础和方向指引。

## 2. PoC 范围与目标

本次 PoC 将聚焦于验证系统的核心感知与知识库构建能力，特别是桌面Agent捕获屏幕、调用图像识别/检索模块、生成/存储结构化分析结果的端到端流程。

### 2.1 PoC 要验证的关键功能和技术点

*   桌面截图捕获的可靠性与效率 (使用 mss/Pillow)。
*   零样本图像识别模块中多模型的协同工作流程 (SAM2, GroundingDINO, BLIP, CLIP, PaddleOCR) 的有效性：验证能否从真实的桌面截图中成功识别出 UI 元素（按钮、输入框、图标、文本块等），并生成边界框、分类、描述和 OCR 文本。
*   `ImageAnalysisResult` 结构化数据生成：验证能否将各模型的分析结果正确整合到 `image_dataclass` 定义的结构中。
*   图像检索模块基于 CLIP 特征和向量数据库（Milvus/Qdrant）实现相似度搜索的可行性：验证能否用新截图检索到知识库中已存储的相似截图。
*   向量数据库存储 `ImageAnalysisResult`（包含向量及元数据）的能力和查询性能。
*   核心决策逻辑：“检索优先”策略的初步验证：能否根据检索相似度决定是复用已知结果还是进行零样本识别。
*   桌面Agent协调各模块工作流程的能力。

### 2.2 PoC 验证的端到端场景

本次 PoC 将重点验证用户需求中提到的核心场景：

#### 场景一：桌面截图分析与知识库核心流程验证

描述: 此场景模拟用户触发截图操作或Agent主动捕获截图。系统需要对截图进行分析，并根据“检索优先”的策略决定流程。如果截图是全新的（在知识库中找不到高相似度的记录），则触发完整的零样本识别流程，并将分析结果存储到知识库中。

*   要验证的核心流程:
    *   桌面Agent捕获屏幕截图。
    *   桌面Agent调用图像检索模块，使用截图的全局特征（例如 CLIP Encoder 输出）在向量数据库中执行相似度搜索。
    *   模拟知识库中未找到高相似度匹配项的情况（例如，使用全新的应用界面截图）。
    *   桌面Agent根据检索结果（低相似度）判定为未知场景。
    *   桌面Agent调用零样本图像识别模块。
    *   零样本图像识别模块按内部流程协同调用 SAM2, GroundingDINO, BLIP, CLIP, PaddleOCR 对截图进行分析：
        *   SAM2 自动生成掩码。
        *   GroundingDINO 根据预设 Prompt 检测 UI 元素。
        *   PaddleOCR 识别文本。
        *   将检测/分割结果与 OCR 文本进行关联。
        *   对检测/分割出的区域使用 BLIP 生成描述。
        *   对检测/分割出的区域使用 CLIP 进行零样本分类（例如分类到“button”, “input”等类别）。
        *   提取全局和区域 CLIP 特征向量。
    *   将所有分析结果整合，生成结构化的 `ImageAnalysisResult` 对象（符合 `image_dataclass` 定义）。
    *   桌面Agent调用数据处理与存储模块，将 `ImageAnalysisResult` 存储到向量数据库中（包括向量和元数据）。
    *   验证存储的 `ImageAnalysisResult` 是否正确反映了截图内容，并能否通过 ID 从数据库中获取。

*   输入:
    *   桌面Agent捕获的屏幕截图（可以是 PNG 或其他格式）。
    *   零样本识别模型所需的 Prompt 列表（例如 GroundingDINO 的目标词汇，CLIP 的分类类别）。
    *   向量数据库连接配置。

*   处理流程:
    1.  Trigger Screenshot Capture.
    2.  Extract Global Feature of captured image.
    3.  Query Vector Database with Global Feature.
    4.  Evaluate Retrieval Results -> Determine as Unknown Scene (due to low similarity).
    5.  Call Zero-Shot Recognition Module.
    6.  Inside Zero-Shot Module: Run SAM2, GroundingDINO, PaddleOCR, BLIP, CLIP sequentially or in parallel based on design.
    7.  Integrate Model Outputs.
    8.  Build `ImageAnalysisResult` object.
    9.  Call Data Storage Module to perist `ImageAnalysisResult` in Vector Database.
    10. Verify Data Storage and retrieval by ID.

*   预期输出:
    *   一个成功的 `ImageAnalysisResult` 对象，包含对有意义的 UI 元素（如按钮、文本框、图标、文本块）的边界框、分类（如“button”、“text input”）、描述和 OCR 文本。
    *   `ImageAnalysisResult` 对象成功存储到向量数据库中的记录 ID。
    *   通过记录 ID 从向量数据库中成功检索出完整的 `ImageAnalysisResult` 对象。

#### 场景二：基于识别结果的简化GUI交互引导

描述: 此场景模拟一个上层的 GUI Agent（或一个简单的用户指令）需要与屏幕上的UI元素进行交互（例如，点击一个按钮，在搜索框输入文本）。系统需要捕获屏幕，并通过识别（零样本或检索）定位指令相关的UI元素，最终输出可供GUI Agent执行的定位信息。此场景重点验证从用户指令到定位目标 UI 元素的能力。

*   要验证的核心流程:
    *   桌面Agent捕获屏幕截图。
    *   桌面Agent按其核心逻辑处理截图（此处假设流程已经经过了检索优先判断，无论是否命中知识库，最终都有一个 `ImageAnalysisResult`）。如果是未知场景，则执行零样本识别生成新的分析结果；如果是已知场景，则从知识库加载已有的分析结果。
    *   模拟一个简单的用户指令，例如：“点击‘文件’菜单”或“在搜索框输入‘测试’”。
    *   根据用户指令和最新生成的 `ImageAnalysisResult`，模拟图像推理模块的简化能力：在一个简化的规则或逻辑中，根据指令中的关键词（“文件”，“搜索框”）在 `ImageAnalysisResult` 的 `region_instances` 列表中查找匹配的 UI 元素。匹配可以基于 `class_name`, `description`, 或 `ocr_text` 字段。
    *   例如，对于“点击‘文件’菜单”，查找 `class_name` 包含“menu”或“button”，且 `ocr_text` 或 `description` 包含“文件”的 `ImageInstance`。
    *   对于“在搜索框输入‘测试’”，查找 `class_name` 包含“input”或“text input”，且附近（或已知布局）是搜索区域的 `ImageInstance`。
    *   如果找到匹配的 UI 元素，提取其边界框 (`bbox`) 和唯一的 `ImageInstance` ID。
    *   输出找到的目标元素的定位信息（`ImageInstance` ID 和 BBox）。
    *   此场景不涉及模拟实际的鼠标点击或键盘输入操作，只验证通过视觉感知定位 UI 元素的能力。

*   输入:
    *   桌面Agent捕获的屏幕截图。
    *   用户的简化指令（字符串格式，例如 "点击 文件 菜单"）。
    *   一个已生成的或加载的 `ImageAnalysisResult` 对象。

*   处理流程:
    1.  Trigger Screenshot Capture.
    2.  Desktop Agent processes screenshot (Retrieval or/and Zero-Shot Recognition). Assume a `ImageAnalysisResult` is obtained/loaded.
    3.  Receive Simulated User Instruction (Text).
    4.  Simulate Simplified Image Inference: Parse Instruction and match keywords against `ImageAnalysisResult.region_instances` fields (`class_name`, `description`, `ocr_text`).
    5.  Identify Target `ImageInstance` and its `bbox`.
    6.  Output Found Target Information.

*   预期输出:
    *   如果成功匹配到指令对应的 UI 元素，输出该元素的 `ImageInstance` ID 和其在原图中的边界框 (`bbox`)。
    *   如果未匹配到，输出查找失败的信息。

此场景与系统集成策略文档中关于赋能GUI Agent部分的关联在于，它验证了图像知识库如何提供结构化的 UI 元素识别结果 (`ImageAnalysisResult`)，以及推理模块（在此处为简化逻辑）如何利用这些信息，根据用户指令定位目标，从而生成 GUI Agent 执行操作所需的定位信息（目标元素的 ID 和位置）。

## 3. 技术栈选型

基于已有的技术调研和模块设计，PoC 将采用以下关键技术、模型和工具：

*   桌面截图: `mss` (高性能跨平台)
*   图像预处理: `Pillow` (PIL)
*   零样本分割: `SAM2` (`sam2.sam2_image_predictor.SAM2ImagePredictor` 和 `sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator`)
*   开放域目标检测: `GroundingDINO` (根据 GroundingDINO Technical Report 中提及的预训练模型加载)
*   OCR: `PaddleOCR` (或其 Python 库)
*   图像区域描述: `BLIP` 或 `BLIP-2` (`transformers` 库加载预训练模型)
*   零样本区域分类/特征提取: `CLIP` (`transformers` 库加载预训练模型)
*   结构化数据定义: Python `dataclasses` (`image_dataclass`)
*   向量数据库: Qdrant (或 Milvus，选择其中一个易于本地部署且支持元数据存储的作为 PoC 数据库)
*   向量索引技术: HNSW (由选定的向量数据库提供)
*   图像推理 (简化模拟): Python 内置字典查找和字符串匹配逻辑。
*   编程语言: Python

## 4. 实施计划与关键步骤

PoC 实施将分为主要两个阶段，对应上述两个场景。

阶段一：桌面截图分析与知识库核心流程 (场景一)

*   时间表: 预计 X 周 (根据资源和复杂度调整)
*   关键步骤:
    1.  环境搭建: 安装所有依赖库 (mss, Pillow, PyTorch, Transformers, PaddleOCR, sam2库, GroundingDINO库, Qdrant/Milvus)。下载所有预训练模型权重。
    2.  桌面Agent原型: 编写 Python 脚本实现：
        *   使用 `mss` 捕获全屏截图，转换为 Pillow Image 或 NumPy 数组。
        *   实现调用 Zero-Shot Recognition Module (见步骤 3)。
        *   实现调用 Vector Database (见步骤 4)。
        *   实现基本逻辑：捕获 -> 特征提取 -> 检索 -> (低相似度) -> 调用零样本识别 -> 生成并存储 `ImageAnalysisResult` -> 验证存储。
    3.  零样本识别模块原型: 编写 Python 类或函数封装：
        *   加载并初始化 SAM2, GroundingDINO, BLIP, CLIP, PaddleOCR 模型。
        *   实现截图输入处理 (`bytes` -> array/Image)。
        *   实现内部协同流程：调用各模型、处理中间结果（如 GroundingDINO BBox 与 SAM2 Mask 的关联）。
        *   实现 `ImageAnalysisResult` 对象的构建。
    4.  向量数据库模块原型:
        *   根据选定的向量数据库 (Qdrant/Milvus)，搭建单机实例。
        *   编写 Python 客户端代码，实现 Collection (表) 的创建，定义向量字段和元数据字段（对应 `ImageAnalysisResult` 结构）。
        *   实现 `store_analysis_result` 方法：从 `ImageAnalysisResult` 提取全局/区域向量和元数据，并插入数据库。
        *   实现 `get_analysis_result_by_id` 方法：根据 ID 从数据库获取记录并重构 `ImageAnalysisResult` 对象。
        *   实现 `vector_search` 方法：根据查询向量执行 HNSW 相似度搜索，返回 `RetrievalResultEntry` 列表。
    5.  CLIP 特征提取: 在 Desktop Agent 或 Zero-Shot Recognition Module 中集成 CLIP Image Encoder 调用，提取截图的全局特征。
    6.  端到端测试 (未知场景): 使用不同应用程序的截图作为输入，运行 Desktop Agent 原型，验证零样本识别流程是否能成功生成 `ImageAnalysisResult` 并将其存储到向量数据库。验证存储的记录内容是否正确。

阶段二：基于识别结果的简化GUI交互定位 (场景二)

*   时间表: 预计 Y 周 (在阶段一基础上，可与阶段一并行部分工作)
*   关键步骤:
    1.  复用阶段一能力: 利用阶段一已实现的截图捕获、零样本识别/知识库加载 (`ImageAnalysisResult` 获取)能力。
    2.  图像推理模拟原型: 编写 Python 函数实现简化推理逻辑：
        *   接收用户指令字符串和 `ImageAnalysisResult` 对象。
        *   解析用户指令，提取关键词。
        *   遍历 `ImageAnalysisResult.region_instances` 列表。
        *   在中对每个 `ImageInstance` 的 `class_name`, `description`, `ocr_text` 字段进行关键词匹配。
        *   如果找到匹配项，返回匹配的 `ImageInstance` ID 和 `bbox`。
    3.  Agent 流程整合: 在 Desktop Agent 原型中整合：
        *   接收用户指令模拟输入。
        *   执行截图捕获、获取 `ImageAnalysisResult`。
        *   调用简化推理逻辑。
        *   输出推理结果（目标 ID 和 BBox）。
    4.  端到端测试 (GUI 定位):
        *   准备带有特定 UI 元素（如“文件”菜单、“搜索框”）的截图。
        *   运行 Agent，输入相应的简化指令（例如，“点击 文件 菜单”）。
        *   验证 Agent 是否能捕获截图、获取分析结果、并通过简化推理逻辑成功定位到期望的 UI 元素，并输出其 ID 和 BBox。

## 5. 预期成果与交付物

*   可运行的 PoC 原型代码库: 实现上述两个场景核心流程的 Python 代码。代码应包含：
    *   Desktop Agent 主体逻辑。
    *   Zero-Shot Recognition Module 关键模型调用与协同逻辑。
    *   Vector Database Client 基本存取与检索逻辑。
    *   Simplified Inference Module（简单的匹配逻辑）。
    *   `image_dataclass` 定义。
*   测试报告: 记录测试过程中遇到的问题、模型性能初步评估（例如，识别出的 UI 元素数量、准确性定性评估）、端到端流程耗时记录（例如，从截图到生成 `ImageAnalysisResult` 的时间，检索耗时）、向量数据库存取基本性能。
*   演示脚本/视频: 用于展示 PoC 原型的关键功能和流程。
*   遇到的关键问题列表及初步解决方案探讨: 记录 PoC 过程中识别的技术难题和对此的初步分析及可能的解决方向，为后续开发提供输入。

## 6. 评估标准与成功指标

*   场景一 (截图分析与知识库):
    *   零样本识别效果: 对已知数据集内的 UI 元素，模型组合能否成功识别出主要 UI 元素 (按钮、输入框、文本块) 并在 `ImageAnalysisResult` 中有正确的 BBox, 类别和描述 (定性评估)。例如，在一张包含 5 个按钮、2 个输入框的截图上，成功识别出 >= 4 个按钮和 >= 1 个输入框。
    *   `ImageAnalysisResult` 结构的完整性: 生成的 `ImageAnalysisResult` 对象能否正确包含原图及识别区域的 BBox, class_name, description, ocr_text, vector 等字段。
    *   向量数据库存储与检索: `ImageAnalysisResult` 数据能否成功插入数据库。通过其 ID 能否在合理的时间内检索出完整对象。
    *   端到端耗时 (零样本路径): 从截图捕获到 `ImageAnalysisResult` 生成及存储完成的初步耗时（例如，在指定硬件环境下 <= 10 秒）。
*   场景二 (GUI交互定位):
    *   目标元素定位成功率: 对于指令中的典型 UI 元素（例如，带有特定文本的按钮、输入框），简化推理逻辑根据 `ImageAnalysisResult` 能否成功找到并输出正确的 BBox 和 ID。例如，对 10 个包含明确目标元素的截图和指令对，定位成功率 >= 80%。
    *   输出格式正确性: 输出的定位信息是否包含目标的 `ImageInstance` ID 和 BBox，并符合预期格式。

## 7. 所需资源

*   人力:
    *   1-2 名具备 Python 开发经验的工程师，熟悉深度学习框架（PyTorch, Transformers）、计算机视觉、数据库操作。
*   硬件:
    *   具备至少 8GB (推荐 16GB+) 显存的 NVIDIA GPU 的开发机器，用于加速深度学习模型推理。
    *   足够的内存和磁盘空间用于存储模型权重和少量测试数据。
*   软件:
    *   Python 3.8+ 环境。
    *   所需 Python 库 (见技术栈选型)。
    *   选定的向量数据库单机版实例 (如 Qdrant Docker 镜像或 Milvus Standalone)。
*   数据:
    *   少量真实的桌面应用程序截图作为测试输入（涵盖不同类型的 UI 元素和布局）。
    *   （可选）少量为测试零样本识别效果而人工标注的区域信息或类别标签（用于评估，而非训练）。

## 8. 风险与缓解措施

| 风险                                   | 描述                                                                                                                                                                                             | 初步缓解措施                                                                                                                                                                                                                                                               |
| :------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 模型推理性能慢                     | 深度的零样本模型（SAM2, BLIP-2, GroundingDINO）计算量大，在桌面端实现端到端快速分析可能存在挑战。                                                                                                                          | 优先在具备 GPU 的环境下测试和部署。评估使用模型量化（如 INT8）、剪枝等技术。考虑使用 ONNX Runtime 或 TensorRT 进行模型推理优化。如果本地性能不足，考虑 PoC 阶段将部分模型推理部署到云端或高性能服务器（但 PoC 目标是验证本地可行性，此为替代方案）。                                                                                             |
| 零样本识别准确性不佳                 | 现有零样本模型在通用领域表现良好，但在特定桌面 UI 元素的细粒度识别上（特别是无文本标签的图标、自定义控件、复杂嵌套布局）可能准确率不如预期。                                                                                               | 仔细设计 GroundingDINO 和 CLIP 的 Prompt 列表，使用更贴近 UI 语境的词汇。利用多模型结果进行后处理融合，提高鲁棒性（例如，BBox 与 Mask 关联，结合 OCR 文本）。在 PoC 评估阶段对识别结果进行定性分析，识别模型的薄弱点，为后续优化（如少量数据微调、Prompt Engineering 优化）提供依据。                                                                             |
| 数据结构复杂度与数据库集成             | `ImageAnalysisResult` 结构包含多层次信息和不同类型数据（向量、字符串、整数、嵌套列表）。在向量数据库中存储和检索这种复杂结构可能需要细致的设计。                                                                                             | 严格按照 `image_dataclass` 定义，使用支持复杂元数据和嵌套结构的向量数据库（Qdrant, Milvus 在调研中显示支持较好）。测试不同存储 scheme（例如，一个 Collection 存全局，一个存区域）的方便性与性能。                                                                                               |
| 环境配置与模型依赖安装               | 深度学习模型依赖和环境配置（如 CUDA 版本匹配、库冲突）可能复杂且耗时。SAM2 和 GroundingDINO 库依赖较多。                                                                                                                                  | 使用虚拟环境/Conda 管理依赖。尽量使用 Poetry 或 Pipenv 等工具锁定依赖版本。提前研究模型库的安装文档和依赖要求。考虑使用 Docker 容器封装模型推理环境，简化部署。                                                                                                                                      |
| 图像推理模拟的局限性                 | PoC 中对图像推理的模拟非常简化，实际的推理需要更复杂的逻辑和强大的模型。PoC 中无法全面验证真实推理能力。                                                                                                                                   | 明确 PoC 阶段只验证“基于识别结果定位目标元素”这一特定推理子任务。后续在系统开发阶段再引入成熟的多模态推理模型（如更强的 VLM）和更复杂的推理逻辑。PoC 目标是验证输入（`ImageAnalysisResult`）与输出（目标 BBox/ID）之间的基本关联可行性。                                                                                     |
| 缺乏标注数据进行定量评估             | 零样本项目缺乏大规模标注数据集， PoC 阶段对识别准确率的评估主要依靠定性分析，难以进行严格的定量评估。                                                                                                                                 | 在 PoC 阶段接受定性评估和基于少量内部构造的测试用例进行评估。验证流程是否成功，输出是否基本符合预期。如果需要更严格的评估，可以先对少量关键场景和 UI 元素进行人工标注，用于计算精确率、召回率等指标，但这会增加 PoC 工作量。                                                                                                 |
| 向量数据库性能在大规模数据下的表现未知 | PoC 阶段使用少量数据，向量数据库的索引和检索性能在大规模知识库下的表现未知。                                                                                                                               | 选择在大规模数据方面有良好口碑和实测报告的向量数据库（如 Milvus, Qdrant 的 HNSW 索引）。 PoC 阶段验证基本功能和初步性能。在后续开发中持续进行性能测试和调优，必要时进行索引参数优化、分布式部署或更换数据库。                                                                                               |

本 PoC 计划为实现桌面端智能代理系统的核心感知和知识库功能提供了详细的路线图。通过成功完成此 PoC，我们将验证关键技术栈的可行性，识别潜在的挑战，并为系统的全面开发奠定坚实基础。
# BiReG 双语提示词50组：已确认精简版

版本：`bireg_development_pairs_50_v1`；整理日期：2026-09-23；状态：作者已同意排除指定9组。

## 规模与用途

保留50组中英配对，即100条语言文本；这是50个场景的两种语言版本。保留原始BIREG编号，编号不连续属于正常情况。原59组归档未改写；中英文原文及来源字段未修改。

本版本用于后续模板和工作流程开发。它记录本次对历史59组的整理决定，不自动证明这些编号就是原稿当时所指的50组。此次未运行新的LLM评分、修改模板或重新生图。

|原分组|保留组数|
|---|---:|
|short|17|
|long|26|
|spatial_relation|3|
|regional_contrast|4|

全部15组标记为must_keep的论文示例均保留。

## 已排除的9组

|编号|中文原文|排除理由|
|---|---|---|
|BIREG-001|实验室中科学家观察试管反应|科学家—试管—实验室，一个观察活动；不要求独立区域或显式相对位置。英文未明确“反应”。|
|BIREG-002|无人机飞越城市高楼|无人机飞越高楼，单一飞行主体；与008同属飞行器场景，016增加计数和编队要求。|
|BIREG-003|工程师调试控制台|工程师操作控制台；“调试”难仅凭静态图像确认，独立区域要求少。|
|BIREG-005|卫星环绕地球运行|卫星—地球存在环绕关系，但无需多实体属性分配；单张图像难区分运行与静止位置。|
|BIREG-008|战斗机高速飞过蓝天|战斗机—蓝天，主要是单主体飞行和背景；“高速”不是直接可验证的静态关系。|
|BIREG-009|坦克在沙地中前行|坦克—沙地，单一主体移动；特定物体与场景有多样性价值，但区域规划信息较少。|
|BIREG-017|少女撑伞站在雪中|与012主体、伞相同，雨中走改为雪中站；不是同义重复，若重视天气／动作对照可保留。|
|BIREG-019|战士手持长枪站立|单名战士持枪站立；007增加人数和场景，026增加前后关系，三者中本条约束最少。|
|BIREG-020|机器人搬运货物|单机器人搬货；004保留相同动作并增加三台数量和工厂环境，优先作为精简候选。|

排除依据包含语义相近及区域／组合约束较少，并非九组都是完全重复文本；未根据模型评分选择。

## 使用前的文本复核点

原文保持不变，以下问题保留待复核：014的“对坐”在英文中未明确；039撒网的动作阶段存在中英表述差异；042英文增加太阳相对雪山位置；046“对襟”与cross-collared不完全对应。确认后应另存文本修订版本。

## 完整保留清单

### BIREG-004

**中文：** 三台机器人在工厂搬运物料

**English:** Three robots carry materials in a factory.

原分组：short；论文保留标记：否；来源：PDF-004 / 0627_short。

### BIREG-006

**中文：** 办公桌上放着一份文件，两个水杯，三只钢笔，两盆花。

**English:** On the desk, there is a document, two cups, three fountain pens, and two potted plants.

原分组：short；论文保留标记：是；来源：PDF-006 / 0627_short。

### BIREG-007

**中文：** 两位军人持枪站在哨所

**English:** Two armed soldiers stand guard at a checkpoint.

原分组：short；论文保留标记：否；来源：PDF-007 / 0627_short。

### BIREG-010

**中文：** 导弹升空腾起火焰

**English:** A missile rises into the air, trailing flames.

原分组：short；论文保留标记：否；来源：PDF-010 / 0627_short。

### BIREG-011

**中文：** 通信兵背着设备穿越丛林

**English:** A communications soldier carries equipment through the jungle.

原分组：short；论文保留标记：否；来源：PDF-011 / 0627_short。

### BIREG-012

**中文：** 少女撑伞走在雨中

**English:** A girl walks in the rain with an umbrella.

原分组：short；论文保留标记：否；来源：PDF-012 / 0627_short。

### BIREG-013

**中文：** 古桥之上站着一匹白马

**English:** A white horse stands on an ancient bridge.

原分组：short；论文保留标记：否；来源：PDF-013 / 0627_short。

### BIREG-014

**中文：** 三位书生对坐而谈

**English:** Three scholars sit and talk together.

原分组：short；论文保留标记：否；来源：PDF-014 / 0627_short。

### BIREG-015

**中文：** 实验室里两名学生讨论电路图

**English:** Two students discuss a circuit diagram in the lab.

原分组：short；论文保留标记：否；来源：PDF-015 / 0627_short。

### BIREG-016

**中文：** 四架无人机编队飞行

**English:** Four drones fly in formation.

原分组：short；论文保留标记：否；来源：PDF-016 / 0627_short。

### BIREG-018

**中文：** 四只猫蹲在窗台

**English:** Four cats sit on the windowsill.

原分组：short；论文保留标记：否；来源：PDF-018 / 0627_short。

### BIREG-021

**中文：** 书桌上有一盏台灯

**English:** A desk lamp is on the desk.

原分组：short；论文保留标记：否；来源：PDF-021 / 0627_short。

### BIREG-022

**中文：** 天台左侧有五棵松树

**English:** Five pine trees stand on the left side of the rooftop.

原分组：short；论文保留标记：是；来源：PDF-022 / 0627_short。

### BIREG-023

**中文：** 两个女孩在街角说话

**English:** Two girls are talking on the street corner.

原分组：short；论文保留标记：是；来源：PDF-023 / 0627_short。

### BIREG-024

**中文：** 屏幕显示红色警报

**English:** The screen displays a red alert.

原分组：short；论文保留标记：否；来源：PDF-024 / 0627_short。

### BIREG-025

**中文：** 两辆汽车停在门口

**English:** Two cars are parked at the entrance.

原分组：short；论文保留标记：否；来源：PDF-025 / 0627_short。

### BIREG-026

**中文：** 士兵端枪站在堡垒前

**English:** A soldier stands in front of a fortress, holding a rifle at the ready.

原分组：short；论文保留标记：否；来源：PDF-026 / 0627_short。

### BIREG-027

**中文：** 清晨，村庄被薄雾笼罩，几位村民在溪边洗衣，远处炊烟袅袅，屋顶上几只鸟儿停歇，狗躺在门前打盹。

**English:** In the early morning, a village is shrouded in mist. Several villagers are washing clothes by the creek. Smoke curls from chimneys in the distance, birds rest on rooftops, and a dog naps by the door.

原分组：long；论文保留标记：是；来源：PDF-027 / 0627_long。

### BIREG-028

**中文：** 音乐厅中，观众安静就坐，指挥缓缓举起指挥棒，乐队成员屏息等待，大厅穹顶镶有金饰，灯光柔和暖黄。

**English:** Inside the concert hall, the audience sits quietly. The conductor slowly raises the baton as the orchestra members hold their breath. The dome ceiling is adorned with golden trim, bathed in warm yellow light.

原分组：long；论文保留标记：否；来源：PDF-028 / 0627_long。

### BIREG-029

**中文：** 实验舱内，宇航员身穿白色太空服在零重力状态下操作面板，墙壁上粘贴着中英文标识，舱外则可见蔚蓝地球悬浮在深邃星空中。

**English:** Inside a space module, an astronaut in a white spacesuit operates a control panel in zero gravity. The walls are marked with Chinese and English labels. Outside the window, the blue Earth floats against the vast starry sky.

原分组：long；论文保留标记：否；来源：PDF-029 / 0627_long。

### BIREG-030

**中文：** 森林深处，两位猎人一前一后追踪野鹿，前方灌木丛中闪现鹿影，地面铺满落叶，头顶阳光从枝叶缝隙中洒下，显现出斑驳光影。

**English:** Deep in the forest, two hunters, one following the other, track a wild deer. The deer is glimpsed in the bushes ahead. Fallen leaves cover the ground, and sunlight filters through gaps in the foliage overhead, casting dappled patterns of light and shadow.

原分组：long；论文保留标记：否；来源：PDF-030 / 0627_long。

### BIREG-031

**中文：** 展览馆中央展台上，一台机械臂正在展示自动书写功能，围观者中有人拍照，有人交谈，展台后方是互动大屏，屏幕实时展示笔迹轨迹与识别结果，整体空间科技感十足。

**English:** At the center of the exhibition hall, a robotic arm demonstrates automated writing. Among the onlookers, some take photos while others engage in conversation. Behind the display, an interactive screen shows real-time stroke tracking and recognition output, creating a futuristic atmosphere.

原分组：long；论文保留标记：否；来源：PDF-031 / 0627_long。

### BIREG-032

**中文：** 港湾夜色下，灯塔微光穿透浓雾，一艘渔船缓缓驶回，甲板上水手忙着整理渔网，海面反射出船灯光影，岸边堆放着空筐和冰块，海鸥掠过夜空。

**English:** In the nighttime harbor, faint lighthouse light pierces the dense fog. A fishing boat slowly returns as sailors organize fishing nets on the deck. The sea reflects the boat's lights; empty baskets and blocks of ice are piled on the shore, while seagulls sweep across the night sky.

原分组：long；论文保留标记：否；来源：PDF-032 / 0627_long。

### BIREG-033

**中文：** 古代演武场上，年轻武者身着白衣正在比武，左侧一位评审捋须而立，右侧观众席上围坐文人墨客，有人记录技艺，有人低声讨论，远处鼓声渐响，战旗迎风招展，黄沙飞扬中，一招一式尽显精妙。

**English:** On an ancient martial-arts ground, a young fighter dressed in white competes in a duel. To the left, a judge stands stroking his beard; in the spectator stands on the right, scholars and poets sit together. Some record the techniques while others converse quietly. In the distance, drums begin to sound and battle flags wave in the wind. Amid swirling yellow sand, every move displays finesse and precision.

原分组：long；论文保留标记：否；来源：PDF-033 / 0627_long。

### BIREG-034

**中文：** 现代艺术馆里，一位母亲带着孩子在观看三维光影装置，旁边有讲解员用平板演示作品背后的数据逻辑，天花板吊灯映出彩色图案，地板反光如镜，游客络绎不绝，背景播放着低频氛围音乐。

**English:** Inside a modern art museum, a mother and her child observe a 3D light-and-shadow installation. Nearby, a docent uses a tablet to demonstrate the data logic behind the work. Ceiling lights cast colorful patterns, the floor reflects like a mirror, visitors stream through the space, and low-frequency ambient music plays in the background.

原分组：long；论文保留标记：否；来源：PDF-034 / 0627_long。

### BIREG-035

**中文：** 冬季夜晚的城市场景中，街道两侧店铺灯火辉煌，人流穿梭不息，有外卖员在奔跑，有情侣牵手驻足橱窗前，一位小提琴手在街角演奏，地面反射着灯光与人影，远处地铁口人群汹涌而出，高楼LED屏幕滚动播放最新新闻，城市喧嚣与人文情感交织。

**English:** On a winter night in the city, shops on both sides of the street glow brightly as crowds move constantly through the scene. A food-delivery courier runs past, a couple holds hands and pauses before a shop window, and a violinist performs at a street corner. The ground reflects lights and human silhouettes. In the distance, crowds surge out of a subway entrance, while LED screens on tall buildings scroll through the latest news. Urban bustle and human emotion intertwine.

原分组：long；论文保留标记：否；来源：PDF-035 / 0627_long。

### BIREG-036

**中文：** 山地公路上，一支徒步队伍正在攀登陡坡，前方领队高举旗帜指引方向，中段队员拍照打气，后方一名队员在检查鞋带，路旁标志提醒注意落石，山腰处有无人机低空巡视，山谷中响起队员口号，阳光穿透云雾洒在山脊之上，层次丰富，动静结合。

**English:** On a mountain road, a hiking team climbs a steep slope. At the front, the leader raises a flag to indicate the direction. In the middle, team members take photographs and encourage one another, while a member at the rear checks their shoelaces. A roadside sign warns of falling rocks, a drone patrols at low altitude along the mountainside, and the team's chants echo through the valley. Sunlight penetrates the clouds and mist, falling across the ridge and creating a layered scene that combines motion and stillness.

原分组：long；论文保留标记：否；来源：PDF-036 / 0627_long。

### BIREG-037

**中文：** 影视拍摄棚内，导演站在监视器前紧盯画面，助理忙于对接拍摄计划，化妆师在一旁为演员补妆，道具组正调整背景布景与灯光角度，场务在地上贴摄像机移动轨迹标线，天花板吊灯被精确调整至指定色温，演员在舞台上排练关键动作，背景循环播放配乐，整组人员各司其职，节奏紧凑而有序，构建出一场现代视听艺术的幕后图景。

**English:** Inside a film studio, the director watches the monitor intently while an assistant coordinates the shooting schedule. A makeup artist touches up an actor nearby, and the props team adjusts the background set and lighting angles. Stage crew tape markings for the camera's movement path onto the floor. Overhead lights are precisely adjusted to the specified color temperature, while an actor rehearses a key movement on stage and background music plays repeatedly. Each crew member performs a distinct role in a fast-paced but orderly behind-the-scenes scene of modern audiovisual art.

原分组：long；论文保留标记：否；来源：PDF-037 / 0627_long。

### BIREG-038

**中文：** 春日庙会人潮涌动，沿街彩旗招展，商贩叫卖声此起彼伏，舞狮队伍正通过主干道，观众高举手机录影，孩童在糖画摊前排队，传统手艺人与游客交流制作技巧，古装表演者穿梭其中与人合影，远处有戏台正在演出昆曲，观众席边设有志愿服务台与医疗点，广播循环播报注意事项，文化氛围浓厚且井然有序。

**English:** At a bustling spring fair, colorful flags line the streets while vendors shout to sell their goods. A lion dance team parades down the main road as spectators hold up phones to record. Children queue at the sugar painting stall. Traditional artisans chat with visitors about craft techniques, and performers in historical costumes pose for photos. In the distance, an open-air stage hosts a Kunqu opera. Nearby are volunteer service stations and medical tents. Loudspeakers broadcast announcements, creating a lively yet orderly cultural atmosphere.

原分组：long；论文保留标记：否；来源：PDF-038 / 0627_long。

### BIREG-039

**中文：** 清晨薄雾弥漫，远山重叠若隐若现，水面平静如镜。画面偏左，一位中年渔夫站立在乌篷船头，身穿褐色粗布衣，头戴斗笠，左手高高扬起渔网，右臂蓄力展开，撒网动作有力自然，网绳在空中展开如伞，动态定格清晰有张力。网下水面泛起圈圈波纹。画面右下角，两只白鹅正在滩涂岸边缓步前行，鹅影清晰映入浅水，岸边有细碎石子、芦苇与水草点缀，动静结合，整体构图均衡。

**English:** In the early morning, mist lingers in the air, with layered distant mountains faintly visible. The water surface is calm like a mirror. Slightly to the left of the composition, a middle-aged fisherman stands at the bow of a black-awning boat, dressed in coarse brown clothing and wearing a bamboo hat. His left hand is raised high, while his right arm is drawn back with strength, poised to cast the net in a powerful and natural motion. The fishing net spreads in midair like an umbrella, creating a dynamic moment full of tension. Below the net, ripples radiate across the water’s surface. In the lower right corner of the image, two white geese are slowly walking along the muddy shore. Their reflections are clearly visible in the shallow water, while scattered pebbles, reeds, and aquatic grasses decorate the bank. The composition blends motion and stillness, achieving an overall sense of balance and harmony.

原分组：long；论文保留标记：是；来源：PDF-039 / 0705_supplement。

### BIREG-040

**中文：** 竹林小径上，一位书生手持书卷边走边读，阳光透过竹叶斑驳洒落。右侧山石间野菊丛生，一只松鼠顺着竹枝一跃而下，轻巧落在书生前方的石板上，尾巴微微上翘，仿佛也在聆听书中之意。

**English:** On a bamboo-lined path, a scholar strolls slowly, reading from a scroll in his hand, while sunlight filters through the leaves, casting dappled shadows across the ground. Among the rocks on the right, wild chrysanthemums bloom in clusters. A squirrel leaps gracefully from a bamboo branch, landing lightly on the stone path ahead of the scholar, its tail raised slightly, as if listening intently to the words being read.

原分组：long；论文保留标记：是；来源：PDF-040 / 0705_supplement。

### BIREG-041

**中文：** 崖顶风大，一名红衣侠客立于悬崖边，背后披风翻飞，远方雷电划过夜空，崖下惊飞群鸟，山谷中回荡回音。

**English:** Strong winds sweep the clifftop as a swordsman in red stands at the edge, his cape fluttering behind him. In the distance, lightning cuts across the night sky. Below the cliff, a flock of startled birds takes flight, while echoes reverberate through the valley.

原分组：long；论文保留标记：否；来源：PDF-041 / 0705_supplement。

### BIREG-042

**中文：** 清晨，日照金山映照高原，一位藏族老阿妈身披暗红藏袍，站在草坡上，左手持转经筒，右手持念珠，神情庄重虔诚。她身后右侧不远处，一头乌黑牦牛静立草坡边缘，抬头望向前方。雪山脚下，一片湖水波光粼粼，倒映着雪峰与佛塔的光辉，湖边是一座白色曲登佛塔，沐浴在朝阳中，香烟袅袅升腾，氛围庄严而宁静。

**English:** In the early morning, golden sunlight illuminates the plateau as the sun rises behind snow-capped mountains. An elderly Tibetan woman stands on a grassy slope, wearing a dark red traditional Tibetan robe. She holds a prayer wheel in her left hand and prayer beads in her right, with a solemn and devout expression. Not far behind her to the right, a black yak stands quietly at the edge of the slope, gazing forward. At the foot of the snow-capped mountains lies a shimmering lake reflecting the brilliance of the peaks and a white chorten (Tibetan stupa) on its shore. Bathed in the morning light, the chorten emits curling incense smoke, creating a sacred and tranquil atmosphere.

原分组：long；论文保留标记：是；来源：PDF-042 / 0705_supplement。

### BIREG-043

**中文：** 春日庙会的戏台上，一位京剧演员身穿凤冠霞帔，正演绎《贵妃醉酒》，她面容端庄，水袖翻飞，台下小朋友好奇张望，一位画糖人的艺人正在一旁专注制作糖人。

**English:** On the stage of a spring temple fair, a Peking Opera performer wearing an ornate phoenix crown and embroidered ceremonial robe performs The Drunken Concubine. Her expression is dignified and her water sleeves flutter through the air. Children watch curiously below the stage, while a sugar-painting artisan nearby concentrates on making a sugar figure.

原分组：long；论文保留标记：否；来源：PDF-043 / 0705_supplement。

### BIREG-044

**中文：** 川剧变脸演员，黑金戏服，红披风，张口吐火，烈焰炽热，单膝跪地，火光照脸，冬季戏台，红灯笼高挂，布幔装饰，观众围观，儿童惊讶，大人拍照，民俗演出，高清电影风，强光冷暖对比，浓厚中国文化气息

**English:** A Sichuan Opera face-changing performer, dressed in a black-and-gold costume and a red cloak, kneels on one knee and breathes blazing fire from his mouth. The flames illuminate his face. The performance takes place on a winter stage decorated with hanging red lanterns and draped curtains. Spectators gather around: children watch in amazement while adults take photographs. The scene depicts a traditional folk performance in a high-definition cinematic style, with strong contrast between warm and cool lighting and a rich Chinese cultural atmosphere.

原分组：long；论文保留标记：是；来源：PDF-044 / 0705_supplement。

### BIREG-045

**中文：** 夜幕下的夜市灯火通明，一位戴口罩的小贩正用手扇着烧烤炉火，铁签翻动间滋滋作响。画面左侧摊位堆满糖葫芦、龙须糖、麦芽饼，包装纸泛红光。右侧一位女孩伸手指着糖人，母亲半蹲为她讲解。

**English:** Under the night sky, a brightly lit night market is bustling with activity. A masked vendor fans the fire of a barbecue grill as iron skewers turn and sizzle. On the left, a stall is piled with candied hawthorn, dragon's-beard candy, and malt cakes, their wrapping paper reflecting reddish light. On the right, a young girl points at a sugar figurine while her mother crouches beside her and explains it.

原分组：long；论文保留标记：否；来源：PDF-045 / 0705_supplement。

### BIREG-046

**中文：** 午后阳光洒落朱红宫墙，枝影斜映，宫道静谧。一位宫女静立石栏前，身着湖蓝对襟长衫，白绣细致，手扶栏杆神情恬静，身旁宫灯微亮。另一位宫女身着浅紫长袍，自画面远端缓步而行，长发盘髻，步履轻盈，身影被夕光拉长，拐角隐入深廊，画面留白中富有动感。

**English:** Afternoon sunlight falls on vermilion palace walls, casting diagonal shadows of branches across a tranquil palace path. A palace maid stands quietly before a stone railing, wearing a lake-blue, cross-collared long robe with fine white embroidery. One hand rests on the railing, her expression calm, while a palace lantern glows faintly beside her. Another maid in a light-purple robe walks slowly from the far end of the scene. Her long hair is coiled into a bun, her steps are light, and the evening glow stretches her silhouette as she disappears around a corner into a deep corridor. The open space gives the composition a sense of movement.

原分组：long；论文保留标记：否；来源：PDF-046 / 0705_supplement。

### BIREG-047

**中文：** 渔船靠岸，码头弥漫着咸腥气。一位妇女提着渔筐走在湿滑石板上，脚穿草屐，筐里装着弹跳的黄鳝与泥鳅。画面远端，一条条挂着编号的渔网在海风中飘动，天边一线残阳染红水面。

**English:** Fishing boats dock as the pier fills with a salty, fishy smell. A woman carrying a fish basket walks across wet, slippery stone slabs in straw clogs. The basket contains thrashing rice-field eels and loaches. In the distance, numbered fishing nets flutter in the sea breeze, while a narrow band of the setting sun dyes the water red along the horizon.

原分组：long；论文保留标记：否；来源：PDF-047 / 0705_supplement。

### BIREG-048

**中文：** 晨雾中，几名少年在竹林间练剑，身法灵动。画面中央，一人正作“起手式”，长剑微颤，目光犀利。左侧老者拄杖观望，足下落叶纷飞。竹枝遮光形成斑驳光影，远处鸟鸣穿林而过。

**English:** In the morning mist, several youths practice swordsmanship in a bamboo grove with agile movements. At the center, one assumes an opening stance, his long sword trembling slightly and his gaze sharp. On the left, an elderly man watches while leaning on a cane as fallen leaves swirl around his feet. Bamboo branches filter the light into dappled patterns, while birdsong carries through the distant forest.

原分组：long；论文保留标记：否；来源：PDF-048 / 0705_supplement。

### BIREG-049

**中文：** 清晨草原上，阳光穿透云层洒下金色光束，左上角一名身穿红衣的小女孩正在追逐一只跳跃的小羊羔，动作灵动，童趣十足。画面中央偏右，一位牧民身穿棕色蒙古袍，长发随风飘扬，正巡视着密集的羊群，姿态稳健。羊群毛发蓬松，错落排布，层次自然。远处蒙古包伫立，背景云卷云舒，阳光映衬出高对比的光影氛围。前景羊群毛发细节清晰，草丛随风轻摆，整体构图疏密有致，风格为高清电影风。

**English:** In an early-morning grassland, golden sunlight pierces the clouds and casts beams across the scene. In the upper left, a little girl dressed in red playfully chases a leaping lamb, her movements lively and full of childlike joy. Toward the center-right, a herder in a brown Mongolian robe, his long hair flowing in the wind, calmly surveys a dense flock of sheep with a steady posture. The sheep have fluffy coats and are arranged in natural layers. In the distance, yurts stand beneath rolling clouds, while sunlight creates high-contrast patterns of light and shadow. In the foreground, the sheep's wool is rendered in crisp detail and the grass sways gently in the breeze. The overall composition balances dense and sparse elements and is presented in a high-definition cinematic style.

原分组：long；论文保留标记：否；来源：PDF-049 / 0728_cover_tests。

### BIREG-050

**中文：** 夜晚灯笼高挂的城市夜市中，一位身穿深灰色连帽卫衣的少年正面注视镜头，脸部呈现出左右融合的特殊面孔：左半边为红底黑白勾勒的传统京剧脸谱，凤眼浓眉，线条刚劲；右半边为金属质感的仿生机械人脸，轮廓锐利，结构复杂，蓝色电子眼明亮炽烈，闪烁着科技光泽。少年神情坚定冷峻，卫衣沾有细微颜料痕迹，体现创作或行动感。背景为灯光斑斓的夜市街道，黄橙色灯笼自上而下延伸，光影层次分明，氛围热烈而略带神秘。

**English:** In a city night market illuminated by hanging lanterns, a teenage boy in a dark-gray hoodie faces the camera directly. His face has a distinctive fused appearance: the left half is painted with traditional Peking Opera facial makeup on a red base, outlined in bold black and white, with phoenix-shaped eyes and thick eyebrows; the right half is a metallic bionic face with sharp contours, complex structures, and a bright blue electronic eye that glows with a technological sheen. The boy's expression is resolute and stern, and his hoodie bears faint paint marks suggesting creative work or action. Behind him lies a colorful night-market street, with yellow-orange lanterns extending downward from above. The layered lighting creates a lively yet slightly mysterious atmosphere.

原分组：long；论文保留标记：否；来源：PDF-050 / 0728_cover_tests。

### BIREG-051

**中文：** 昏黄灯光洒落在展馆中，古代兵马俑列阵两侧，陶土质感沉稳，神态肃穆，排列整齐，历史氛围浓厚。中央伫立一位银白色仿生机器人，身高约两米，金属外壳反射冷光，关节结构清晰，面部为无表情仿生面罩，科技感强烈。机器人姿态挺拔，静静注视前方，与兵马俑神态相似，材质对比强烈，构成古今交汇的视觉张力。顶部为金属穹顶结构，光源柔和偏暖，玻璃护栏外有孩童与观众驻足远观，孩童仰头指认，神情惊叹。科技与传统在暖光与历史厚重感中交融呈现。

**English:** Warm dim light falls across an exhibition hall, where ancient terracotta warriors stand in orderly formations on both sides, their earthen textures subdued and expressions solemn. At the center stands a silver-white bionic robot about two meters tall, its metallic shell reflecting cool light, with clearly articulated joints and an expressionless biomimetic face mask. The robot stands upright and looks forward, echoing the posture of the warriors while creating a strong material contrast between antiquity and technology. A metal dome rises overhead under soft warm lighting. Beyond the glass railing, children and other visitors stop to observe; the children look up and point in amazement. Technology and tradition merge within the warm light and weight of history.

原分组：long；论文保留标记：否；来源：PDF-051 / 0728_cover_tests。

### BIREG-052

**中文：** 城市文化广场的一面巨型汉字书法墙上，笔走龙蛇的行草字映入眼帘。墙前，一位年长书法家正在以毛笔临摹古文，气韵生动，笔锋有力。墙的另一侧，几位年轻街头艺术家正用数字喷枪绘制涂鸦字体，流光溢彩、造型夸张，实时投影技术将笔画动态映在墙面上。书法与涂鸦并列展示，形成古今文字艺术的强烈对比，围观人群驻足拍照，场面热烈。

**English:** A giant wall of Chinese calligraphy dominates an urban cultural plaza, covered with energetic cursive characters. In front of it, an elderly calligrapher copies a classical text with a brush, using vigorous and expressive strokes. On the other side of the wall, several young street artists use digital spray tools to create colorful, exaggerated graffiti lettering, while real-time projection maps the moving strokes onto the wall. Traditional calligraphy and graffiti are displayed side by side, creating a striking contrast between ancient and contemporary writing arts as an enthusiastic crowd stops to take photographs.

原分组：long；论文保留标记：否；来源：PDF-052 / 0728_cover_tests。

### BIREG-053

**中文：** 一位小男孩站在一位老人的前方。

**English:** A young boy stands in front of an elderly man.

原分组：spatial_relation；论文保留标记：是；来源：PDF-U01 / 20260405_rule_based。

### BIREG-054

**中文：** 一只白猫趴在一只黑狗的头上。

**English:** A white cat lies on top of a black dog's head.

原分组：spatial_relation；论文保留标记：是；来源：PDF-U02 / 20260405_rule_based。

### BIREG-055

**中文：** 一只白猫趴在一只黑狗的头上，黑狗站在一个木箱前面。

**English:** A white cat lies on top of a black dog's head, while the black dog stands in front of a wooden box.

原分组：spatial_relation；论文保留标记：是；来源：PDF-U03 / 20260405_rule_based。

### BIREG-056

**中文：** 幽冥的世界，漆黑的天空中漂浮着幽灵的身影，地面上是死寂的灰烬，恶魔的低语在空中回荡，给人一种绝望与恐惧的感觉。

**English:** In the underworld, ghostly figures float beneath a pitch-black sky. The ground is covered with lifeless ashes, and demonic whispers echo through the air, creating a feeling of despair and fear.

原分组：regional_contrast；论文保留标记：是；来源：PAPER-F4-01 / paper_only。

### BIREG-057

**中文：** 天堂的光辉从云层中洒下，金色的光束照耀大地，洁白的建筑矗立在云端，天使们飞翔在蓝天中，四周弥漫着宁静与神圣。

**English:** Heavenly radiance pours through the clouds, and golden beams illuminate the earth. White buildings rise above the clouds, while angels fly across the blue sky. The entire scene is filled with serenity and holiness.

原分组：regional_contrast；论文保留标记：是；来源：PAPER-F4-02 / paper_only。

### BIREG-058

**中文：** 森林中的树木高耸入云，古老的藤蔓爬满了树干，空气中弥漫着草木的芳香，隐秘的生物在林间穿行，神秘而宁静。

**English:** Trees tower into the clouds in the forest, and ancient vines cover their trunks. The air is filled with the fragrance of plants and trees, while hidden creatures move through the woods, creating a mysterious and tranquil atmosphere.

原分组：regional_contrast；论文保留标记：是；来源：PAPER-F4-03 / paper_only。

### BIREG-059

**中文：** 荒原上，风沙肆虐，干裂的土地上没有生物的踪影，远处是一片荒芜的景象，只有枯萎的草丛和石堆。

**English:** Across the wasteland, wind-driven sand sweeps over cracked earth with no trace of living creatures. The distant landscape is desolate, containing only withered grass and piles of stones.

原分组：regional_contrast；论文保留标记：是；来源：PAPER-F4-04 / paper_only。

## 文件说明

- `bireg_development_pairs_50_v1.jsonl`：实际使用的50组清单，保留原字段和编号。
- `excluded_pairs_9.jsonl`：排除原文、编号及理由。
- `selection_record.json`：版本、保留／排除编号和哈希。
- `bireg_prompt_pairs_v1.jsonl`：原始59组副本。
- `review_59.jsonl`：此前逐条分析快照，其中proposal_only状态指当时分析阶段；当前选择状态以selection_record.json为准。
- `build_subset.py`：标准库复算脚本，在本目录运行可重新生成50组清单及记录。

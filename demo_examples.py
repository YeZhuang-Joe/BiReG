DEMO_CASES = {
    "palace_two_maids": {
        "desc": "Dual-subject palace corridor scene with explicit foreground-background separation.",
        "prompt":"In warm afternoon sunlight, a vermilion palace wall fills the background, with tree branches casting strong oblique shadows across its surface, creating a high-contrast cinematic lighting effect.In the left foreground, a palace maid stands as the primary subject, positioned near a carved stone railing. She wears a lake-blue cross-collared robe with delicate white embroidery. Her hand gently rests on the railing, her posture calm and composed, and her face softly illuminated, forming the main visual focus of the scene.In the right background, a second palace maid must be present at a much smaller scale, located deep along the palace path. She wears a light purple robe and walks slowly toward a distant corridor. Her figure is elongated by the evening light and slightly blurred due to depth, clearly indicating spatial distance.Both figures must appear simultaneously and remain spatially separated, with no overlap. The stone pathway and railing create strong perspective lines that guide the viewer’s gaze from the foreground figure to the distant figure, forming a clear depth hierarchy and narrative contrast between stillness and motion.",
        "split_ratio": "0.35;0.40,0.6,0.4;0.25",
        "regional_prompt":"A towering vermilion palace wall dominates the background, illuminated by warm afternoon sunlight. Soft, elongated shadows of tree branches fall diagonally across the wall, creating a rhythmic pattern of light and shade. The atmosphere is calm, historical, and slightly nostalgic, with subtle variations in wall texture and light diffusion.BREAK A palace maid stands quietly beside a carved white marble railing. She is dressed in a lake-blue cross-collared long robe with delicate white embroidery on the collar and sleeves. Her posture is upright yet relaxed, one hand gently resting on the railing. Her expression is calm and introspective, gazing slightly into the distance. A traditional palace lantern beside her emits a soft warm glow, subtly contrasting with the cooler tones of her clothing. She is the primary subject, clearly in focus, with sharp detail and natural lighting.BREAK Another palace maid in a light purple robe walks slowly away from the viewer toward a corridor corner. Her figure is smaller and positioned deeper in the scene, slightly out of focus due to depth of field. Her long hair is styled in a neat bun, and her movement is graceful. The warm evening light stretches her shadow across the ground, emphasizing motion and spatial depth. She is partially turning into a shadowed corridor, creating a sense of narrative continuation.BREAK A stone-paved palace pathway extends into the distance, with clear perspective lines guiding the viewer’s eye toward the background. The surface shows fine texture details, subtle wear, and warm light reflections. Long shadows stretch across the ground, reinforcing the time of day and enhancing depth perception. The composition leaves slight open space, contributing to a dynamic yet balanced visual flow.",
        "config": {
            "batch_size": 1,
            "base_ratio": 0.2,
            "num_inference_steps": 40,
            "height": 768,
            "width": 1536,
            "seed": 1234,
            "guidance_scale": 7.5,
            "negative_prompt": ""
        }
    },
    "scholar_squirrel_bamboo": {
        "desc":"",
        "prompt": "竹林小径上，一位书生手持书卷边走边读，阳光透过竹叶斑驳洒落。右侧山石间野菊丛生，一只松鼠顺着竹枝轻巧落在书生前方的山石上，尾巴微微上翘，仿佛也在聆听书中之意。",
        "split_ratio": "0.25;0.55,0.6,0.4;0.20",
        "regional_prompt": "茂密的绿色竹叶层层叠叠，占据了画面的上方。清晨或午后的阳光穿过竹叶的缝隙，形成一道道清晰的光束，斑驳地洒落在下方的景物上。光影交错，营造出一种幽静、深邃且充满生机的竹林氛围。BREAK 一位身穿传统长袍的书生正缓步走在竹林小径上。他手中紧握着一卷展开的书卷，目光低垂，全神贯注地阅读，仿佛周围的喧嚣都与他无关。阳光斑驳地洒在他的肩膀和书卷上，增加了画面的质感和立体感。他的姿态儒雅，透出浓厚的书卷气。BREAK 在右侧的山石与竹林之间，一只灵动的松鼠站在书生前方的石板上上，蓬松的尾巴微微上翘。它的目光似乎投向上方的书生，仿佛被书生的读书声所吸引，想要一探究竟。这一动态与书生的静态形成了鲜明的对比。BREAK 书生的下半身隐没在竹林的阴影中，脚下是一条铺满落叶的石板小径。在书生前方的山石上，正是松鼠即将落下的位置。画面右侧的岩石缝隙中，几丛金黄色的野菊正在盛开，点缀了以绿色为主调的画面，增添了一抹亮色和野趣。",
        "config": {
            "batch_size": 1,
            "base_ratio": 0.2,
            "num_inference_steps": 30,
            "height": 1024,
            "width": 1536,
            "seed": 1234,
            "guidance_scale": 4.5,
            "negative_prompt": ""
        }
    }

}

mmlu_task_names = [
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_physics',
    'electrical_engineering',
    'astronomy',
    'anatomy',
    'abstract_algebra',
    'machine_learning',
    'clinical_knowledge',
    'global_facts',
    'management',
    'nutrition',
    'marketing',
    'professional_accounting',
    'high_school_geography',
    'international_law',
    'moral_scenarios',
    'computer_security',
    'high_school_microeconomics',
    'professional_law',
    'medical_genetics',
    'professional_psychology',
    'jurisprudence',
    'world_religions',
    'philosophy',
    'virology',
    'high_school_chemistry',
    'public_relations',
    'high_school_macroeconomics',
    'human_sexuality',
    'elementary_mathematics',
    'high_school_physics',
    'high_school_computer_science',
    'high_school_european_history',
    'business_ethics',
    'moral_disputes',
    'high_school_statistics',
    'miscellaneous',
    'formal_logic',
    'high_school_government_and_politics',
    'prehistory',
    'security_studies',
    'high_school_biology',
    'logical_fallacies',
    'high_school_world_history',
    'professional_medicine',
    'high_school_mathematics',
    'college_medicine',
    'high_school_us_history',
    'sociology',
    'econometrics',
    'high_school_psychology',
    'human_aging',
    'us_foreign_policy',
    'conceptual_physics',
]

# --------------------------------------------- mmlu groups
mmlu_humanities = [
    'formal_logic', 'high_school_european_history', 'high_school_us_history',
    'high_school_world_history', 'international_law', 'jurisprudence',
    'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy',
    'prehistory', 'professional_law', 'world_religions'
]

mmlu_stem = [
    'abstract_algebra', 'anatomy', 'astronomy', 'college_biology',
    'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_physics', 'computer_security', 'conceptual_physics',
    'electrical_engineering', 'elementary_mathematics', 'high_school_biology',
    'high_school_chemistry', 'high_school_computer_science',
    'high_school_mathematics', 'high_school_physics', 'high_school_statistics',
    'machine_learning'
]

mmlu_social_science = [
    'econometrics', 'high_school_geography',
    'high_school_government_and_politics', 'high_school_macroeconomics',
    'high_school_microeconomics', 'high_school_psychology', 'human_sexuality',
    'professional_psychology', 'public_relations', 'security_studies',
    'sociology', 'us_foreign_policy'
]

mmlu_other = [
    'business_ethics', 'clinical_knowledge', 'college_medicine',
    'global_facts', 'human_aging', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'nutrition',
    'professional_accounting', 'professional_medicine', 'virology'
]

mmlu_all = mmlu_humanities + mmlu_stem + mmlu_social_science + mmlu_other
assert set(mmlu_all) == set(mmlu_task_names)

mmlu_groups = {
    'mmlu_all': mmlu_all,
    'mmlu_humanities': mmlu_humanities,
    'mmlu_stem': mmlu_stem,
    'mmlu_social_science': mmlu_social_science,
    'mmlu_other': mmlu_other,
}

# --------------------------------------------- m_mmlu info

LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(
    ',')
LANGS = [
    'ar',
    'de',
    'es',
    'fr',
    'ru',
    'zh',
    'zht',
]

# copy from https://gitlab.pjlab.org.cn/wujiang/compass-internal/-/blob/wj/mkqa/configs/datasets/mmlu_m/m_mmlu_prompts.py
LANG_TO_INSTRUCTIONS = {
    'BASIC': {
        'en': 'There is a single choice question about {task_name}. Answer the question by replying A, B, C or D.',
        # 'ar': 'هناك سؤال اختيار من متعدد حول {task_name}. أجب عن السؤال بالرد بـ أ، ب، ج، أو د.',
        'de': 'Es gibt eine Single-Choice-Frage zu {task_name}. Beantworten Sie die Frage, indem Sie mit A, B, C oder D antworten.',
        'ru': 'Вопрос с выбором одного варианта ответа по теме {task_name}. Ответьте на вопрос, выбрав A, B, C или D.',
        'zh': '以下是关于{task_name}的单项选择题，请直接回答正确答案A、B、C或D。',
        'zht': '以下是關於{task_name}的單項選擇題，請直接回答正確答案A、B、C或D。',
    },
    'REFUSE' : {
        'en': "There is a single choice question about {task_name}. If you know the answer, please directly respond with the correct answer A, B, C, or D. If you do not know the answer, please respond with \"I don't know.\".",
        'zh': '以下是关于{task_name}的单项选择题，如果你知道答案，请直接回答正确答案A、B、C或D，如果不知道答案，请回答“我不知道”。',
        'zht': '以下是關於{task_name}的單項選擇題，如果你知道答案，請直接回答正確答案A、B、C或D，如果不知道答案，請回答「我不知道」。',
    },
} # yapf: disable

LANG_TO_QUESTION_PREFIX = {
    'en': 'Question',
    'de': 'Frage',
    'es': 'Pregunta',
    'fr': 'Question',
    'ja': '質問',
    'ru': 'Вопрос',
    'th': 'คำถาม',
    'zh': '问题',
    'zht': '問題',
}

LANG_TO_ANSWER_PREFIX = {
    'en': 'Answer',
    'de': 'Antwort',
    'es': 'Respuesta',
    'fr': 'Réponse',
    'ja': '答え',
    'ru': 'Ответ',
    'th': 'คำตอบ',
    'zh': '答案',
    'zht': '答案',
}

mmlu_task_names = {
    'college_biology': {
        'en': 'college biology',
        'ar': 'الأحياء الجامعية',
        'de': 'College-Biologie',
        'es': 'biología universitaria',
        'fr': 'biologie universitaire',
        'ru': 'биология в колледже',
        'zh': '大学生物学',
        'zht': '大學生物學'
    },
    'college_chemistry': {
        'en': 'college chemistry',
        'ar': 'كيمياء الكلية',
        'de': 'College-Chemie',
        'es': 'química universitaria',
        'fr': 'chimie universitaire',
        'ru': 'химия в колледже',
        'zh': '大学化学',
        'zht': '大學化學'
    },
    'college_computer_science': {
        'en': 'college computer science',
        'ar': 'كلية علوم الحاسب الآلي',
        'de': 'Hochschulinformatik',
        'es': 'informática universitaria',
        'fr': "l'informatique au collège",
        'ru': 'информатика в колледже',
        'zh': '大学计算机科学',
        'zht': '大學計算機科學'
    },
    'college_mathematics': {
        'en': 'college mathematics',
        'ar': 'الرياضيات الجامعية',
        'de': 'Hochschulmathematik',
        'es': 'matemáticas universitarias',
        'fr': 'mathématiques universitaires',
        'ru': 'высшая математика',
        'zh': '大学数学',
        'zht': '大學數學'
    },
    'college_physics': {
        'en': 'college physics',
        'ar': 'فيزياء الكلية',
        'de': 'Hochschulphysik',
        'es': 'física universitaria',
        'fr': 'la physique au collège',
        'ru': 'физика в колледже',
        'zh': '大学物理',
        'zht': '大學物理'
    },
    'electrical_engineering': {
        'en': 'electrical engineering',
        'ar': 'الهندسة الكهربائية',
        'de': 'Elektrotechnik',
        'es': 'ingeniería eléctrica',
        'fr': 'génie électrique',
        'ru': 'электротехника',
        'zh': '电气工程',
        'zht': '電氣工程'
    },
    'astronomy': {
        'en': 'astronomy',
        'ar': 'علم الفلك',
        'de': 'Astronomie',
        'es': 'astronomía',
        'fr': 'astronomie',
        'ru': 'астрономия',
        'zh': '天文',
        'zht': '天文'
    },
    'anatomy': {
        'en': 'anatomy',
        'ar': 'التشريح',
        'de': 'Anatomie',
        'es': 'anatomía',
        'fr': 'anatomie',
        'ru': 'анатомия',
        'zh': '解剖',
        'zht': '解剖'
    },
    'abstract_algebra': {
        'en': 'abstract algebra',
        'ar': 'الجبر المجرد',
        'de': 'abstrakte Algebra',
        'es': 'álgebra abstracta',
        'fr': 'algèbre abstraite',
        'ru': 'абстрактная алгебра',
        'zh': '抽象代数',
        'zht': '抽象代數'
    },
    'machine_learning': {
        'en': 'machine learning',
        'ar': 'التعلُّم الآلي',
        'de': 'maschinelles Lernen',
        'es': 'aprendizaje automático',
        'fr': 'apprentissage automatique',
        'ru': 'машинное обучение',
        'zh': '机器学习',
        'zht': '機器學習'
    },
    'clinical_knowledge': {
        'en': 'clinical knowledge',
        'ar': 'المعرفة السريرية',
        'de': 'klinische Kenntnisse',
        'es': 'conocimientos clínicos',
        'fr': 'connaissances cliniques',
        'ru': 'клинические знания',
        'zh': '临床知识',
        'zht': '臨牀知識'
    },
    'global_facts': {
        'en': 'global facts',
        'ar': 'حقائق عالمية',
        'de': 'globale Fakten',
        'es': 'hechos globales',
        'fr': 'faits mondiaux',
        'ru': 'глобальные факты',
        'zh': '全球事实',
        'zht': '全球事實'
    },
    'management': {
        'en': 'management',
        'ar': 'الإدارة',
        'de': 'Management',
        'es': 'gestión',
        'fr': 'gestion',
        'ru': 'управление',
        'zh': '管理',
        'zht': '管理'
    },
    'nutrition': {
        'en': 'nutrition',
        'ar': 'التغذية',
        'de': 'Ernährung',
        'es': 'nutrición',
        'fr': "l'alimentation",
        'ru': 'питание',
        'zh': '营养学',
        'zht': '營養學'
    },
    'marketing': {
        'en': 'marketing',
        'ar': 'التسويق',
        'de': 'Marketing',
        'es': 'marketing',
        'fr': 'marketing',
        'ru': 'маркетинг',
        'zh': '市场营销',
        'zht': '市場營銷'
    },
    'professional_accounting': {
        'en': 'professional accounting',
        'ar': 'المحاسبة المهنية',
        'de': 'professionelle Buchhaltung',
        'es': 'contabilidad profesional',
        'fr': 'comptabilité professionnelle',
        'ru': 'профессиональный бухгалтерский учёт',
        'zh': '专业会计',
        'zht': '專業會計'
    },
    'high_school_geography': {
        'en': 'high school geography',
        'ar': 'جغرافية المدرسة الثانوية',
        'de': 'Geographie in der Oberstufe',
        'es': 'geografía de secundaria',
        'fr': 'géographie au lycée',
        'ru': 'география средней школы',
        'zh': '高中地理',
        'zht': '高中地理'
    },
    'international_law': {
        'en': 'international law',
        'ar': 'القانون الدولي',
        'de': 'internationales Recht',
        'es': 'derecho internacional',
        'fr': 'droit international',
        'ru': 'международное право',
        'zh': '国际法',
        'zht': '國際法'
    },
    'moral_scenarios': {
        'en': 'moral scenarios',
        'ar': 'السيناريوهات الأخلاقية',
        'de': 'moralische Szenarien',
        'es': 'escenarios morales',
        'fr': 'scénarios moraux',
        'ru': 'моральные сценарии',
        'zh': '道德情景',
        'zht': '道德情景'
    },
    'computer_security': {
        'en': 'computer security',
        'ar': 'أمن الحاسب الآلي',
        'de': 'Computersicherheit',
        'es': 'seguridad informática',
        'fr': 'sécurité informatique',
        'ru': 'компьютерная безопасность',
        'zh': '计算机安全',
        'zht': '計算機安全'
    },
    'high_school_microeconomics': {
        'en': 'high school microeconomics',
        'ar': 'الاقتصاد الجزئي في المرحلة الثانوية',
        'de': 'Mikroökonomie in der Oberstufe',
        'es': 'microeconomía en secundaria',
        'fr': 'microéconomie au lycée',
        'ru': 'микроэкономика в средней школе',
        'zh': '高中微观经济学',
        'zht': '高中微觀經濟學'
    },
    'professional_law': {
        'en': 'professional law',
        'ar': 'القانون المهني',
        'de': 'Berufsrecht',
        'es': 'derecho profesional',
        'fr': 'droit professionnel',
        'ru': 'профессиональное право',
        'zh': '专业法',
        'zht': '專業法'
    },
    'medical_genetics': {
        'en': 'medical genetics',
        'ar': 'علم الوراثة الطبية',
        'de': 'medizinische Genetik',
        'es': 'genética médica',
        'fr': 'génétique médicale',
        'ru': 'медицинская генетика',
        'zh': '医学遗传学',
        'zht': '醫學遺傳學'
    },
    'professional_psychology': {
        'en': 'professional psychology',
        'ar': 'علم النفس المهني',
        'de': 'Berufspsychologie',
        'es': 'psicología profesional',
        'fr': 'psychologie professionnelle',
        'ru': 'профессиональная психология',
        'zh': '专业心理学',
        'zht': '專業心理學'
    },
    'jurisprudence': {
        'en': 'jurisprudence',
        'ar': 'الفقه',
        'de': 'Rechtssprechung',
        'es': 'jurisprudencia',
        'fr': 'jurisprudence',
        'ru': 'юриспруденция',
        'zh': '判例',
        'zht': '判例'
    },
    'world_religions': {
        'en': 'world religions',
        'ar': 'أديان العالم',
        'de': 'Weltreligionen',
        'es': 'religiones del mundo',
        'fr': 'religions du monde',
        'ru': 'мировые религии',
        'zh': '世界宗教',
        'zht': '世界宗教'
    },
    'philosophy': {
        'en': 'philosophy',
        'ar': 'الفلسفة',
        'de': 'Philosophie',
        'es': 'filosofía',
        'fr': 'philosophie',
        'ru': 'философия',
        'zh': '哲学',
        'zht': '哲學'
    },
    'virology': {
        'en': 'virology',
        'ar': 'علم الفيروسات',
        'de': 'Virologie',
        'es': 'virología',
        'fr': 'virologie',
        'ru': 'вирусология',
        'zh': '病毒学',
        'zht': '病毒學'
    },
    'high_school_chemistry': {
        'en': 'high school chemistry',
        'ar': 'كيمياء المدارس الثانوية',
        'de': 'Oberstufenchemie',
        'es': 'química en secundaria',
        'fr': 'chimie au lycée',
        'ru': 'химия в средней школе',
        'zh': '高中化学',
        'zht': '高中化學'
    },
    'public_relations': {
        'en': 'public relations',
        'ar': 'العلاقات العامة',
        'de': 'Öffentlichkeitsarbeit',
        'es': 'relaciones públicas',
        'fr': 'relations publiques',
        'ru': 'связи с общественностью',
        'zh': '公共关系',
        'zht': '公共關係'
    },
    'high_school_macroeconomics': {
        'en': 'high school macroeconomics',
        'ar': 'الاقتصاد الكلي في المرحلة الثانوية',
        'de': 'Makroökonomie in der Oberstufe',
        'es': 'macroeconomía en secundaria',
        'fr': 'macroéconomie au lycée',
        'ru': 'макроэкономика в средней школе',
        'zh': '高中宏观经济学',
        'zht': '高中宏觀經濟學'
    },
    'human_sexuality': {
        'en': 'human sexuality',
        'ar': 'النشاط الجنسي البشري',
        'de': 'menschliche Sexualität',
        'es': 'sexualidad humana',
        'fr': 'la sexualité humaine',
        'ru': 'человеческая сексуальность',
        'zh': '人类性行为',
        'zht': '人類性行爲'
    },
    'elementary_mathematics': {
        'en': 'elementary mathematics',
        'ar': 'الرياضيات الابتدائية',
        'de': 'elementare Mathematik',
        'es': 'matemáticas elementales',
        'fr': 'mathématiques élémentaires',
        'ru': 'элементарная математика',
        'zh': '初等数学',
        'zht': '初等數學'
    },
    'high_school_physics': {
        'en': 'high school physics',
        'ar': 'فيزياء المدارس الثانوية',
        'de': 'Oberstufenphysik',
        'es': 'física de secundaria',
        'fr': 'physique au lycée',
        'ru': 'физика в средней школе',
        'zh': '高中物理',
        'zht': '高中物理'
    },
    'high_school_computer_science': {
        'en': 'high school computer science',
        'ar': 'علوم الحاسب الآلي في المدارس الثانوية',
        'de': 'Informatik in der Oberstufe',
        'es': 'informática en secundaria',
        'fr': "l'informatique au lycée",
        'ru': 'информатика в средней школе',
        'zh': '高中计算机科学',
        'zht': '高中計算機科學'
    },
    'high_school_european_history': {
        'en': 'high school european history',
        'ar': 'التاريخ الأوروبي في المرحلة الثانوية',
        'de': 'Europäische Geschichte in der Oberstufe',
        'es': 'historia europea de bachillerato',
        'fr': 'histoire européenne au lycée',
        'ru': 'история европы в средней школе',
        'zh': '高中欧洲历史',
        'zht': '高中歐洲歷史'
    },
    'business_ethics': {
        'en': 'business ethics',
        'ar': 'أخلاقيات العمل',
        'de': 'Unternehmensethik',
        'es': 'ética empresarial',
        'fr': 'éthique des affaires',
        'ru': 'деловая этика',
        'zh': '商业道德',
        'zht': '商業道德'
    },
    'moral_disputes': {
        'en': 'moral disputes',
        'ar': 'النزاعات الأخلاقية',
        'de': 'Gewissenskonflikte',
        'es': 'disputas morales',
        'fr': 'conflits moraux',
        'ru': 'моральные споры',
        'zh': '道德争议',
        'zht': '道德爭議'
    },
    'high_school_statistics': {
        'en': 'high school statistics',
        'ar': 'إحصاءات المدارس الثانوية',
        'de': 'Oberschulstatistik',
        'es': 'estadísticas de secundaria',
        'fr': "statistiques de l'enseignement secondaire",
        'ru': 'статистика средней школы',
        'zh': '高中统计',
        'zht': '高中統計'
    },
    'miscellaneous': {
        'en': 'miscellaneous',
        'ar': 'متفرقات',
        'de': 'Verschiedenes',
        'es': 'varios',
        'fr': 'divers',
        'ru': 'разное',
        'zh': '杂项',
        'zht': '雜項'
    },
    'formal_logic': {
        'en': 'formal logic',
        'ar': 'المنطق الشكلي',
        'de': 'formale Logik',
        'es': 'lógica formal',
        'fr': 'logique formelle',
        'ru': 'формальная логика',
        'zh': '形式逻辑',
        'zht': '形式邏輯'
    },
    'high_school_government_and_politics': {
        'en': 'high school government and politics',
        'ar': 'الحكومة والسياسة في المدارس الثانوية',
        'de': 'Regierung und Politik in der Oberstufe',
        'es': 'gobierno y política en secundaria',
        'fr': 'gouvernement et politique au niveau secondaire',
        'ru': 'правительство и политика в средней школе',
        'zh': '高中政府与政治',
        'zht': '高中政府與政治'
    },
    'prehistory': {
        'en': 'prehistory',
        'ar': 'عصور ما قبل التاريخ',
        'de': 'Vorgeschichte',
        'es': 'prehistoria',
        'fr': 'préhistoire',
        'ru': 'предыстория',
        'zh': '史前',
        'zht': '史前'
    },
    'security_studies': {
        'en': 'security studies',
        'ar': 'الدراسات الأمنية',
        'de': 'Sicherheitsstudien',
        'es': 'estudios de seguridad',
        'fr': 'études de sécurité',
        'ru': 'исследования в области безопасности',
        'zh': '安全研究',
        'zht': '安全研究'
    },
    'high_school_biology': {
        'en': 'high school biology',
        'ar': 'علم الأحياء في المرحلة الثانوية',
        'de': 'Biologie für die Oberstufe',
        'es': 'biología de secundaria',
        'fr': 'biologie au lycée',
        'ru': 'биология в средней школе',
        'zh': '高中生物',
        'zht': '高中生物'
    },
    'logical_fallacies': {
        'en': 'logical fallacies',
        'ar': 'المغالطات المنطقية',
        'de': 'logische Irrtümer',
        'es': 'falacias lógicas',
        'fr': 'sophismes logiques',
        'ru': 'логические заблуждения',
        'zh': '逻辑谬误',
        'zht': '邏輯謬誤'
    },
    'high_school_world_history': {
        'en': 'high school world history',
        'ar': 'تاريخ العالم في المرحلة الثانوية',
        'de': 'Weltgeschichte in der Oberstufe',
        'es': 'historia del mundo en secundaria',
        'fr': 'histoire mondiale au lycée',
        'ru': 'всемирная история средней школы',
        'zh': '高中世界史',
        'zht': '高中世界史'
    },
    'professional_medicine': {
        'en': 'professional medicine',
        'ar': 'الطب المهني',
        'de': 'Fachmedizin',
        'es': 'medicina profesional',
        'fr': 'médecine professionnelle',
        'ru': 'профессиональная медицина',
        'zh': '专业医学',
        'zht': '專業醫學'
    },
    'high_school_mathematics': {
        'en': 'high school mathematics',
        'ar': 'الرياضيات في المرحلة الثانوية',
        'de': 'Highschool-Mathematik',
        'es': 'matemáticas de secundaria',
        'fr': "mathématiques de l'enseignement secondaire",
        'ru': 'математика в средней школе',
        'zh': '高中数学',
        'zht': '高中數學'
    },
    'college_medicine': {
        'en': 'college medicine',
        'ar': 'طب الكلية',
        'de': 'Hochschulmedizin',
        'es': 'medicina universitaria',
        'fr': 'médecine universitaire',
        'ru': 'студенческая медицина',
        'zh': '大学医学',
        'zht': '大學醫學'
    },
    'high_school_us_history': {
        'en': 'high school us history',
        'ar': 'تاريخ الولايات المتحدة الأمريكية في المرحلة الثانوية',
        'de': 'high school us geschichte',
        'es': 'historia de ee.uu. en secundaria',
        'fr': 'histoire des états-unis au lycée',
        'ru': 'История США в средней школе',
        'zh': '高中美国历史',
        'zht': '高中美國曆史'
    },
    'sociology': {
        'en': 'sociology',
        'ar': 'علم الاجتماع',
        'de': 'Soziologie',
        'es': 'sociología',
        'fr': 'sociologie',
        'ru': 'социология',
        'zh': '社会学',
        'zht': '社會學'
    },
    'econometrics': {
        'en': 'econometrics',
        'ar': 'الاقتصاد القياسي',
        'de': 'Ökonometrie',
        'es': 'econometría',
        'fr': 'économétrie',
        'ru': 'эконометрика',
        'zh': '计量经济学',
        'zht': '計量經濟學'
    },
    'high_school_psychology': {
        'en': 'high school psychology',
        'ar': 'علم النفس في المدارس الثانوية',
        'de': 'Schulpsychologie',
        'es': 'psicología en la escuela secundaria',
        'fr': 'psychologie au lycée',
        'ru': 'психология в средней школе',
        'zh': '高中心理学',
        'zht': '高中心理學'
    },
    'human_aging': {
        'en': 'human aging',
        'ar': 'شيخوخة الإنسان',
        'de': 'menschliche Alterung',
        'es': 'envejecimiento humano',
        'fr': 'le vieillissement humain',
        'ru': 'старение человека',
        'zh': '人类老化',
        'zht': '人類老化'
    },
    'us_foreign_policy': {
        'en': 'us foreign policy',
        'ar': 'السياسة الخارجية الأمريكية',
        'de': 'us-außenpolitik',
        'es': 'política exterior estadounidense',
        'fr': 'politique étrangère des états-unis',
        'ru': 'внешняя политика США',
        'zh': '美国外交政策',
        'zht': '美國外交政策'
    },
    'conceptual_physics': {
        'en': 'conceptual physics',
        'ar': 'الفيزياء المفاهيمية',
        'de': 'konzeptionelle Physik',
        'es': 'física conceptual',
        'fr': 'physique conceptuelle',
        'ru': 'концептуальная физика',
        'zh': '概念物理',
        'zht': '概念物理'
    }
}

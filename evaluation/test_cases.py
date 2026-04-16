# 20 test cases — same as Phase 1/2 manual tests
# Each case: query, expected_category, budget_max, must_include_keywords, must_not_exceed_budget

TEST_CASES = [
    {
        "id": 1,
        "query": "I need a gaming laptop under $1000",
        "expected_category": "gaming",
        "budget_max": 1000,
        "budget_firm": True,
        "must_include": ["rtx", "gaming", "legion", "ideapad"],
        "description": "Gaming under $1000"
    },
    {
        "id": 2,
        "query": "What is the cheapest Lenovo laptop you have?",
        "expected_category": "all",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["299", "ideapad 1"],
        "description": "Cheapest laptop"
    },
    {
        "id": 3,
        "query": "I want a ThinkPad for business travel, must be under 3 lbs",
        "expected_category": "business",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["thinkpad"],
        "weight_max": 3.0,
        "description": "ThinkPad under 3 lbs"
    },
    {
        "id": 4,
        "query": "Show me all Legion laptops with RTX 4070 or better",
        "expected_category": "gaming",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["4070", "legion"],
        "description": "Legion RTX 4070+"
    },
    {
        "id": 5,
        "query": "I need a laptop for video editing with a good display",
        "expected_category": "creative",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["oled", "rtx"],
        "description": "Video editing display"
    },
    {
        "id": 6,
        "query": "I need a business laptop, budget is $800, no more than that",
        "expected_category": "business",
        "budget_max": 800,
        "budget_firm": True,
        "must_include": ["thinkbook", "thinkpad"],
        "description": "Business $800 firm"
    },
    {
        "id": 7,
        "query": "Gaming laptop, must have RTX 4080 or higher, budget flexible",
        "expected_category": "gaming",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["4080", "4090"],
        "description": "RTX 4080+ gaming"
    },
    {
        "id": 8,
        "query": "I travel a lot, need something under 3 lbs, battery over 12 hours",
        "expected_category": "all",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["thinkpad", "ideapad"],
        "weight_max": 3.0,
        "description": "Travel <3lbs >12hrs"
    },
    {
        "id": 9,
        "query": "Student budget $500, mainly for school work and some light gaming",
        "expected_category": "student",
        "budget_max": 500,
        "budget_firm": True,
        "must_include": ["ideapad", "499", "449", "399", "299"],
        "description": "Student $500"
    },
    {
        "id": 10,
        "query": "I need a 2-in-1 convertible for presentations, under $1000",
        "expected_category": "smb",
        "budget_max": 1000,
        "budget_firm": True,
        "must_include": ["2-in-1", "convertible", "touch", "flex"],
        "description": "2-in-1 under $1000"
    },
    {
        "id": 11,
        "query": "I want a gaming laptop but I absolutely do NOT want AMD processor",
        "expected_category": "gaming",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["intel"],
        "must_not_include": ["amd ryzen", "ryzen 7", "ryzen 9"],
        "description": "Gaming no AMD"
    },
    {
        "id": 12,
        "query": "Looking for a ThinkPad, nothing over 4 lbs, I travel constantly",
        "expected_category": "business",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["thinkpad"],
        "weight_max": 4.0,
        "description": "ThinkPad <4 lbs"
    },
    {
        "id": 13,
        "query": "Budget is firm at $700, do not recommend anything above that",
        "expected_category": "all",
        "budget_max": 700,
        "budget_firm": True,
        "must_include": ["ideapad", "649", "699", "599", "449"],
        "description": "Firm $700 budget"
    },
    {
        "id": 14,
        "query": "I need a laptop for CAD work, must have dedicated professional GPU",
        "expected_category": "workstation",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["p1", "rtx", "workstation", "thinkpad"],
        "description": "CAD professional GPU"
    },
    {
        "id": 15,
        "query": "I prefer Intel over AMD, need a business laptop around $1200",
        "expected_category": "business",
        "budget_max": 1200,
        "budget_firm": False,
        "must_include": ["intel", "thinkpad", "thinkbook"],
        "description": "Intel business ~$1200"
    },
    {
        "id": 16,
        "query": "I need something for video editing with color accurate display",
        "expected_category": "creative",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["oled"],
        "description": "RAG test: semantic video editing"
    },
    {
        "id": 17,
        "query": "Long battery life laptop for all day work without charging",
        "expected_category": "all",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["hrs", "battery"],
        "description": "RAG test: battery life semantic"
    },
    {
        "id": 18,
        "query": "Ultralight laptop for frequent flyers who work on the go",
        "expected_category": "business",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["thinkpad", "lbs"],
        "description": "RAG test: ultralight traveler semantic"
    },
    {
        "id": 19,
        "query": "Best laptop for machine learning and AI development",
        "expected_category": "workstation",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["rtx", "gpu", "ram"],
        "description": "RAG test: ML/AI development"
    },
    {
        "id": 20,
        "query": "I need a laptop under $500 with good battery for online classes",
        "expected_category": "student",
        "budget_max": 500,
        "budget_firm": True,
        "must_include": ["ideapad", "chromebook"],
        "description": "RAG test: student online classes"
    },
]
TEST_CASES_V2 = [
    {
        "id": 21,
        "query": "I need the most powerful laptop you have, money is no object",
        "expected_category": "gaming",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["4090", "3499", "legion 9i"],
        "description": "Max performance no budget"
    },
    {
        "id": 22,
        "query": "I only have $300, what can I get?",
        "expected_category": "student",
        "budget_max": 300,
        "budget_firm": True,
        "must_include": ["299", "ideapad 1"],
        "description": "Minimum budget $300"
    },
    {
        "id": 23,
        "query": "I need a laptop that weighs less than 2.5 lbs",
        "expected_category": "business",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["2.48", "2.35", "x1 carbon", "x13s"],
        "description": "Ultra lightweight <2.5 lbs"
    },
    {
        "id": 24,
        "query": "Show me laptops with OLED display under $1200",
        "expected_category": "creative",
        "budget_max": 1200,
        "budget_firm": True,
        "must_include": ["oled"],
        "description": "OLED under $1200"
    },
    {
        "id": 25,
        "query": "I want something for gaming but it must be under 5 lbs",
        "expected_category": "gaming",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["legion slim", "4.63", "4.41", "4.85"],
        "description": "Gaming lightweight <5 lbs"
    },
    {
        "id": 26,
        "query": "I need a Snapdragon laptop with long battery life",
        "expected_category": "business",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["snapdragon", "28hrs", "x13s"],
        "description": "Snapdragon battery life"
    },
    {
        "id": 27,
        "query": "What ThinkPad has the best battery life?",
        "expected_category": "business",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["28", "snapdragon", "x13s"],
        "description": "Best ThinkPad battery"
    },
    {
        "id": 28,
        "query": "I need a laptop for architecture school, CAD and rendering",
        "expected_category": "workstation",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["rtx", "32gb", "gpu"],
        "description": "Architecture CAD rendering"
    },
    {
        "id": 29,
        "query": "Something portable for coffee shop work, light and good battery",
        "expected_category": "business",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["lbs", "hrs"],
        "description": "RAG: coffee shop portable worker"
    },
    {
        "id": 30,
        "query": "I play competitive FPS games and need high refresh rate display",
        "expected_category": "gaming",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["165hz", "144hz", "240hz"],
        "description": "RAG: competitive FPS high refresh"
    },
    {
        "id": 31,
        "query": "I want a laptop for streaming on Twitch and gaming",
        "expected_category": "gaming",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["rtx", "legion", "32gb"],
        "description": "RAG: streaming and gaming"
    },
    {
        "id": 32,
        "query": "Need a laptop for remote work with good webcam and microphone",
        "expected_category": "business",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["thinkpad", "thinkbook"],
        "description": "RAG: remote work webcam"
    },
    {
        "id": 33,
        "query": "I am a nurse and need a durable laptop for hospital use",
        "expected_category": "business",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["thinkpad", "enterprise", "durable"],
        "description": "RAG: healthcare durable laptop"
    },
    {
        "id": 34,
        "query": "Budget $1500, no AMD, no IdeaPad, need dedicated GPU",
        "expected_category": "gaming",
        "budget_max": 1500,
        "budget_firm": True,
        "must_include": ["intel", "rtx", "legion"],
        "must_not_include": ["amd ryzen", "ideapad"],
        "description": "Multi-constraint: no AMD no IdeaPad"
    },
    {
        "id": 35,
        "query": "I need a laptop that can run SolidWorks smoothly",
        "expected_category": "workstation",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["p1", "rtx", "workstation"],
        "description": "RAG: SolidWorks specific software"
    },
    {
        "id": 36,
        "query": "What is the price of the ThinkPad X1 Carbon Gen 13?",
        "expected_category": "business",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["1899", "x1 carbon"],
        "description": "Factual: specific product price"
    },
    {
        "id": 37,
        "query": "How much does the Legion 9i Gen 9 weigh?",
        "expected_category": "gaming",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["5.95", "legion 9i"],
        "description": "Factual: specific product weight"
    },
    {
        "id": 38,
        "query": "I need a laptop for a 10-year-old for school, very basic tasks",
        "expected_category": "student",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["chromebook", "ideapad", "299", "349"],
        "description": "RAG: child basic school use"
    },
    {
        "id": 39,
        "query": "I want to do 3D animation in Blender, what do you recommend?",
        "expected_category": "creative",
        "budget_max": None,
        "budget_firm": False,
        "must_include": ["rtx", "32gb", "oled"],
        "description": "RAG: Blender 3D animation"
    },
    {
        "id": 40,
        "query": "Need something under $600 with Windows, not Chromebook",
        "expected_category": "student",
        "budget_max": 600,
        "budget_firm": True,
        "must_include": ["windows", "ideapad"],
        "must_not_include": ["chromebook"],
        "description": "Budget $600 Windows only no Chromebook"
    },
]
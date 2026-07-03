const STORAGE_KEY = "ainudgingppr-language";
const DEFAULT_LANGUAGE = "en";
const SUPPORTED_LANGUAGES = ["en", "zh-Hant", "zh-Hans"];
const originalText = new WeakMap();

const simplifiedMap = {
    "與": "与", "續": "续", "費": "费", "遊": "游", "業": "业", "導": "导", "頁": "页",
    "關": "关", "於": "于", "團": "团", "隊": "队", "項": "项", "進": "进", "潛": "潜",
    "響": "响", "資": "资", "計": "计", "劃": "划", "雙": "双", "應": "应", "編": "编",
    "號": "号", "稱": "称", "對": "对", "強": "强", "調": "调", "倫": "伦", "設": "设",
    "為": "为", "們": "们", "場": "场", "選": "选", "擇": "择", "範": "范", "圍": "围",
    "權": "权", "負": "负", "責": "责", "實": "实", "踐": "践", "鳴": "鸣", "謝": "谢",
    "別": "别", "區": "区", "料": "料", "見": "见", "結": "结", "論": "论", "議": "议",
    "並": "并", "觀": "观", "點": "点", "戰": "战", "帶": "带", "來": "来", "討": "讨",
    "傳": "传", "統": "统", "機": "机", "邊": "边", "個": "个", "從": "从", "綠": "绿",
    "企": "企", "駕": "驾", "過": "过", "證": "证", "據": "据", "眾": "众", "協": "协",
    "會": "会", "學": "学", "劉": "刘", "網": "网", "郵": "邮", "總": "总", "體": "体",
    "發": "发", "監": "监", "鍾": "钟", "經": "经", "濟": "济", "員": "员", "決": "决",
    "敗": "败", "興": "兴", "蓋": "盖", "構": "构", "預": "预", "產": "产", "價": "价",
    "數": "数", "碼": "码", "時": "时", "種": "种", "條": "条", "運": "运", "維": "维",
    "顧": "顾", "術": "术", "話": "话", "識": "识", "記": "记", "錄": "录", "費": "费",
    "舉": "举", "獲": "获", "國": "国", "長": "长", "專": "专", "線": "线", "創": "创",
    "處": "处", "庫": "库", "組": "组", "輔": "辅", "獻": "献", "務": "务", "據": "据",
    "導": "导", "態": "态", "嗎": "吗", "這": "这", "處": "处", "顯": "显", "稱": "称",
    "該": "该", "內": "内", "產": "产", "給": "给", "優": "优", "雜": "杂", "較": "较"
};

function simplifyText(text) {
    return text.replace(/[\u3400-\u9fff]/g, (character) => simplifiedMap[character] || character);
}

function updateChineseText(language) {
    document.querySelectorAll(".lang-zh").forEach((element) => {
        const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
        let node = walker.nextNode();

        while (node) {
            if (!originalText.has(node)) {
                originalText.set(node, node.nodeValue);
            }

            const sourceText = originalText.get(node);
            node.nodeValue = language === "zh-Hans" ? simplifyText(sourceText) : sourceText;
            node = walker.nextNode();
        }
    });
}

function setLanguage(language) {
    const selectedLanguage = SUPPORTED_LANGUAGES.includes(language) ? language : DEFAULT_LANGUAGE;
    document.documentElement.lang = selectedLanguage;
    localStorage.setItem(STORAGE_KEY, selectedLanguage);
    updateChineseText(selectedLanguage);

    document.querySelectorAll("[data-lang-button]").forEach((button) => {
        const isActive = button.dataset.langButton === selectedLanguage;
        button.classList.toggle("active", isActive);
        button.setAttribute("aria-pressed", String(isActive));
    });
}

document.addEventListener("DOMContentLoaded", () => {
    const savedLanguage = localStorage.getItem(STORAGE_KEY) || DEFAULT_LANGUAGE;
    setLanguage(savedLanguage);

    document.querySelectorAll("[data-lang-button]").forEach((button) => {
        button.addEventListener("click", () => setLanguage(button.dataset.langButton));
    });
});

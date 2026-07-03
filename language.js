const STORAGE_KEY = "ainudgingppr-language";
const DEFAULT_LANGUAGE = "en";

function setLanguage(language) {
    const selectedLanguage = language === "zh-Hant" ? "zh-Hant" : "en";
    document.documentElement.lang = selectedLanguage;
    localStorage.setItem(STORAGE_KEY, selectedLanguage);

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

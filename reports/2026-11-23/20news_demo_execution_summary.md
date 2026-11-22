# 11/23 20 Newsgroups 特徵擷取執行摘要

已於 2026/11/23（Asia/Taipei）完成真實文件展示，使用程式版本
`801138e0f9bdd1473f4d88dfa0b389643956d92b`。本次沿用已提交的擷取程式，
沒有改寫特徵擷取方法，也沒有執行 PSO、集成或分類效能實驗。

## 一、資料來源與實測結果

資料由 `sklearn.datasets.fetch_20newsgroups()` 取得，`subset='all'`、
`shuffle=True`、`random_state=42`、`remove=()`，來源共 **18,846 篇、20 類**。
沒有使用預先向量化版本。原文 CSV 保留此 API 回傳的文字；詞頻另外經由
專案目前的 `afs_rgmnb.features.document.document_vectorizer` 產生。

| 展示版本 | 文件數 | 實際涵蓋類別 | 觀察到的特徵數 | 非零詞頻項目 nnz | 詞頻總和 |
|---|---:|---:|---:|---:|---:|
| first60（主要交付） | 60 | 19 | 3,689 | 5,866 | 8,201 |
| three_per_class（補充） | 60 | 20 | 3,106 | 5,401 | 7,427 |

`first60` 是固定種子打亂後的前 60 篇，不是未打亂原始檔的前 60 篇；
這次沒有抽到 `soc.religion.christian`，其餘類別各 1–6 篇。補充版從同一
排序依序選取每類三篇，只為展示類別多樣性，**不是正式實驗的平衡抽樣政策**。
完整類別分布與來源索引保存在各自的本機 manifest。

這次 API 語料與學長保存的 **18,828 筆／20 類／83,415 個觀察特徵**計數來源
不同。這是從原始文字重新擷取，不是把學長的計數矩陣還原成文字，也沒有
宣稱重現學長的字典、停用字表或歷史前處理。

## 二、實際前處理順序

1. 原文 CSV 先保留下載文字，包括郵件標頭、引言與簽名；本次未使用
   `--strip-metadata`。
2. 詞頻擷取時，專案 analyzer 嘗試移除符合其 RFC822 判斷規則的初始標頭區塊。
   主版實際辨識並移除 **59/60** 篇的標頭，補充版為 **60/60**；不能宣稱所有
   郵件資訊均已清除。引言與簽名沒有額外移除。
3. 轉成小寫，再以專案的 Unicode 字母／數字斷詞規則分詞；保留單字內 ASCII
   連字號（例如 `real-time`），其他標點作為分隔。單字與數字均可能成為詞項。
4. 移除明確指定的 sklearn `ENGLISH_STOP_WORDS`，再使用 NLTK 的
   **original Porter** 詞幹化。因此輸出 term 是處理後的詞幹，不一定是原文拼字。
5. 以 `document_vectorizer` 的 `fit_transform` 計算出現次數；`min_df=1`、
   不限制最大特徵數，輸出 int32 非負整數，沒有 TF-IDF、二值化或長度正規化。

此次停用字表是已揭露的實作選擇，並非已驗證的學長原始清單。
`feature_index` 是各展示獨立字典的零起始索引；兩個展示的字典不可混用。

## 三、逐筆核對與測試

- 以 `csv.DictReader` 重新讀取兩組 CSV：各有 60 篇原文、60 個唯一樣本 ID，
  原文與詞頻檔的 ID 集合、來源索引、類別代碼及名稱完全一致。
- 對照快取中的真實 API 來源，確認所有選取文章文字、標籤及原文 SHA-256；
  沒有替換成人工文件。每組詞頻列數都與實測 nnz 一致。
- 對兩組各 **60/60 篇**重新呼叫專案 analyzer，用 `Counter` 核對所有詞幹的
  出現次數，並重新透過專案 vectorizer 擷取矩陣，與 CSV 的每個非零值比對。
  全部一致，沒有重複的樣本／特徵列，所有儲存值均為正整數；未列出的值為零。
- 主版有 55 篇、補充版有 56 篇含重複詞項；重複次數均正確，兩版最大詞頻
  都是 22。這裡報告彙總結果，不公開原文、個別樣本 ID 或逐筆詞項。
- 全部 **101 項單元／整合測試通過**，測試本體耗時 **6.259 秒**，無失敗或跳過。
  其中展示程式的三項測試使用人工 fixture；上述 CSV 驗證則使用真實文件，兩者分開記錄。
- 最新遠端 `801138e` 的 [Python 3.10 CI](https://github.com/jgw0915/AFS-RGMNB/actions/runs/45056313428)
  已完成且成功，安裝與 unittest 步驟均成功。公開 API 可確認 run/job 狀態；
  逐行日誌下載回應 403，故不把本機測試日誌說成遠端日誌。
- 論文指紋檢查：12 個參考檔案全部相符，論文工作目錄乾淨。

本機環境：Python **3.12.7**、NumPy **1.26.4**、SciPy **1.11.4**、
scikit-learn **1.4.2**、NLTK **3.9.2**；套件依 `requirements.txt` 安裝。
本機執行與遠端 Python 3.10 CI 是不同的驗證證據。

## 四、實際命令與檔案

於儲存庫根目錄的 PowerShell 執行（既有隔離環境，避免變更全域套件）：

```powershell
git fetch origin
git pull --ff-only origin refactor/full-thesis-framework
& '.\outputs\ci_deps312\Scripts\python.exe' -m pip install -r requirements.txt
& '.\outputs\ci_deps312\Scripts\python.exe' -m pip install -e . --no-deps
& '.\outputs\ci_deps312\Scripts\python.exe' scripts/export_20news_60_demo.py --selection first60 --output outputs/2026-11-23/20news_demo --data-home outputs/2026-11-23/sklearn_cache
& '.\outputs\ci_deps312\Scripts\python.exe' scripts/export_20news_60_demo.py --selection three_per_class --output outputs/2026-11-23/20news_demo_three_per_class --data-home outputs/2026-11-23/sklearn_cache
& '.\outputs\ci_deps312\Scripts\python.exe' outputs/2026-11-23/verify_demo.py
& '.\outputs\ci_deps312\Scripts\python.exe' -m unittest discover -s tests -v
& '.\outputs\ci_deps312\Scripts\python.exe' tools/check_thesis_spec.py
git diff --check
git check-ignore -v outputs/2026-11-23/20news_demo/20news_original_60.csv
```

指定 `--data-home` 使完整下載快取也留在 Git 忽略範圍。伺服器中斷後先確認
沒有完成的 CSV，再續跑，未覆寫已完成的展示。完成的主版程序耗時 **109.963 秒**，
包含來源取得／快取準備、載入、擷取與 CSV 輸出，不是純擷取時間；使用完成快取的
補充版程序耗時 **1.210 秒**。中斷前未完成的時間不計入這兩項。

兩版各自的目錄：

- 主版：`outputs/2026-11-23/20news_demo/`
- 補充版：`outputs/2026-11-23/20news_demo_three_per_class/`

每個目錄包含 `20news_original_60.csv`、`20news_extracted_word_counts_60.csv`、
`demo_manifest.json`。第一份 CSV 每列是一篇文章（原文含換行，不能用檔案行數
當作文章數）；第二份是非零詞頻長格式。用相同 `sample_id` 對照即可。

完整本機查核紀錄位於 `outputs/2026-11-23/`：`independent_validation.json`
（CSV／manifest 雜湊、環境、彙總核對）、`execution_times.json`、
`tests.log`、`thesis_check.json`、`ci_runs.json`、`ci_jobs.json` 及
`verify_demo.py`。輸出與下載快取全部受忽略規則保護，不提交公開 Git。

## 五、限制、候選資料與交付建議

本次僅有 60 篇，兩版實測特徵數也都小於 5,000，**不符合正式研究的
至少 500 筆／5,000 個觀察特徵門檻**。字典共同擬合這 60 篇，只用於展示；
沒有 Dfit／Dsearch／Dtest 切分、正式重複／群組獨立性評估或分類正確率。
不能把此字典或樣本當作正式測試折；後續正式實驗仍只從 Dfit 建立字典。

[三份候選資料比較表](new_dataset_candidates.csv) 已列 Reuters R8、WebKB 四類版、
BBC News 指定詞頻版。表內 7,674／17,387／8、4,199／7,770／4、
2,225／6,167／5（樣本／特徵／類別）是**來源報告值**，本次沒有下載或實測這
三份候選。WebKB 必須保留四類處理版的說明；BBC 的 6,167 維必須核對實際版本。
後續再確認格式、自然類別分布、群組切分與各 Dfit 的觀察維度，不默默平衡資料。

建議提供老師本摘要與候選比較 CSV；原文、詞頻及 manifest 可選擇一整組，透過
學校允許的私人方式交付，避免混用兩版字典。原文可能含個資及著作權內容，
不放公開 GitHub。郵件草稿見 [email_draft_zh-TW.md](email_draft_zh-TW.md)，
目前未寄送。實際展示已完成，沒有資料下載或計數驗證阻礙；正式實驗仍待下一階段。

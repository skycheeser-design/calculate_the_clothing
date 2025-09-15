# calculate_the_clothing

## プロジェクト概要
衣類をメジャーで測ることなく、写真一枚から正確な寸法をテキスト化できる採寸アプリです。身丈・肩幅・身幅・袖丈などの寸法を自動的に算出し、メルカリやECサイトへの出品時にそのまま貼り付けられるテキストとして出力します。

## 特徴・機能
- スマートフォンで撮影した画像をもとに寸法を自動計測
- 黒マーカ認識もしくはAI処理を用いた寸法換算
- 計測結果を画像に重ね書きし、コピー可能なテキストとして出力
- 衣類のカテゴリ（例: Tシャツ, 長袖シャツ, 短パン, ズボン, 任意入力）を選択して結果を分類保存
- 収集した画像をAI学習用データセットとして活用する拡張も視野
- 画像の読み込みやマスク平滑化を共有する `image_utils.py` を同梱

## 必要環境 / 依存パッケージ
- Python 3.10 以上
- iPhoneなどのスマートフォン（将来的にはAndroidにも対応予定）
- opencv-python, pillow-heif, numpy, Pillow

### 日本語フォントについて
寸法テキストを日本語で描画するためには日本語に対応したTrueTypeフォントが必要です。
環境にフォントがインストールされていない場合は、以下のように環境変数
`JP_FONT_PATH` でフォントファイルのパスを指定してください。macOS の場合は
`/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc` や
`/Library/Fonts/Kosugi-Regular.ttf` などを指定できます。

```bash
export JP_FONT_PATH=/path/to/your/NotoSansJP-Regular.otf
```

## インストール方法
1. リポジトリをクローンします。
   ```bash
   git clone https://github.com/ユーザー名/calculate_the_clothing.git
   cd calculate_the_clothing
   ```
2. Pythonパッケージをインストールします。
   ```bash
   pip install opencv-python pillow-heif numpy Pillow
   ```

## 使い方
1. 服をスマホで撮影し、画像ファイルを用意します。
2. スクリプトを実行します。
   ```bash
   python Clothing
   ```
3. カテゴリを選択すると、画像と寸法テキストが `results/<カテゴリ名>/` 以下に保存されます。

共通の画像処理関数は `image_utils.py` にまとめられており、`Clothing` や `measurements.py` からインポートして利用できます。

## プロジェクト構成（例）
```
calculate_the_clothing/
├── Clothing        # 服の寸法を計測するPythonスクリプト
├── image_utils.py  # 画像読み込みやマスク平滑化の共通ユーティリティ
├── README.md       # このファイル
└── results/        # 計測結果が保存されるディレクトリ（実行後に生成）
```

## ライセンス情報
ライセンスは未定です。MITやApache 2.0など、プロジェクトに適したライセンスを選び、`LICENSE` ファイルを追加してください。

## 開発・貢献方法
1. リポジトリをフォーク
2. ブランチを作成
3. 修正をコミット
4. Pull Request を送信

バグ報告や機能提案は Issue にて受け付けています。

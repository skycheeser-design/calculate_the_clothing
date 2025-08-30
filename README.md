# calculate_the_clothing

## プロジェクト概要
衣類をメジャーで測ることなく、写真一枚から正確な寸法をテキスト化できる採寸アプリです。身丈・肩幅・身幅・袖丈などの寸法を自動的に算出し、メルカリやECサイトへの出品時にそのまま貼り付けられるテキストとして出力します。

## 特徴・機能
- スマートフォンで撮影した画像をもとに寸法を自動計測
- 黒マーカ認識もしくはAI処理を用いた寸法換算
- 計測結果を画像に重ね書きし、コピー可能なテキストとして出力
- 衣類のカテゴリ（例: Tシャツ, 長袖シャツ, 短パン, ズボン, 任意入力）を選択して結果を分類保存
- 収集した画像をAI学習用データセットとして活用する拡張も視野

## 必要環境 / 依存パッケージ
- Python 3.10 以上
- iPhoneなどのスマートフォン（将来的にはAndroidにも対応予定）
- opencv-python, pillow-heif, rembg, numpy, Pillow

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
   pip install opencv-python pillow-heif rembg numpy Pillow
   ```

## 使い方
1. 服をスマホで撮影し、画像ファイルを用意します。
2. パイプライン用の CLI を実行します。
   ```bash
   python cli.py path/to/image.jpg
   ```
   `--skip-background` や `--skip-measure` などのフラグを付けることで、
   キャッシュ済みの処理をスキップできます。
3. `cache/` ディレクトリに背景除去後の画像や計測結果が保存され、
   標準出力にも寸法値が表示されます。

## モジュール構成
```
calculate_the_clothing/
├── cli.py                # パイプライン制御用 CLI (メインモジュール)
├── clothing/
│   ├── io.py             # 画像読み込みと HEIC 対応
│   ├── background.py     # マーカ検出と背景除去
│   ├── measure.py        # マスク・スケルトンによる寸法計測
│   └── viz.py            # フォント読み込みと寸法描画
└── tests/                # 単体テスト
```

## ライセンス情報
ライセンスは未定です。MITやApache 2.0など、プロジェクトに適したライセンスを選び、`LICENSE` ファイルを追加してください。

## 開発・貢献方法
1. リポジトリをフォーク
2. ブランチを作成
3. 修正をコミット
4. Pull Request を送信

バグ報告や機能提案は Issue にて受け付けています。

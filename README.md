【サーバー立ち上げ方】

0. 'docker load -i webapp.tar' でwebappイメージをload
1. './setup.sh' でdockerコンテナ起動
2. docker上でshellが起動したら/workspaceディレクトリに移動
3. python app.pyでサーバー起動
4. localhostで "http://0.0.0.0:8000"を開く
5. 画面が表示されたら成功
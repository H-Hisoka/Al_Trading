# Al_Trading

Rules
AE phai push len nhanh rieng cua minh truoc khi t merge vao main cho ae
AE co the push nhanh cua minh, con nhanh main phai tao pull request va dc review truoc khi merge

# Lam sao de lay code sau khi 1 nguoi lam xong
Nguoi A lam xong phan dau, push code len nhanh rieng, va tao pull request
Huy se review va merge no vao main
Nguoi B tiep tuc lam thi, len local,nhảy qua nhánh main, git pull origin main ve, sau do nhay qua nhanh phu de lam tiep và git merge main

# Va lam tiep
đang o nhánh phu
git checkout main
git pull origin main
git checkout <nhánh phụ>
git merge main

# để push code cua minh len
dang o nhanh phu
git add (dung folder)
git commit -m "comment gi do"
git push -u origin (ten nhanh phu) (lan dau)
git push (lan sau)

# lan dau mang code ve
tao folder chua
nhay vao folder
git clone (url github)
nhay vao folder
git pull origin main
git checkout -b (ten nhanh phu moi)
git add .
git commit -m "first commit"
git push -u origin (tennhanh phu cua minh)

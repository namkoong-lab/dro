# ----------------------------
# DRO Documentation Build System
# Version: 1.2
# ----------------------------

# 定义常量
SPHINXOPTS    = -W --keep-going -n
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
LIVEPORT      = 8000
DOCTREES      = $(BUILDDIR)/doctrees

# 文件类型忽略模式
IGNORE_PATTERNS = "*.swp" "*.tmp" "*.bak" "*.pdf"

# 伪目标声明
.PHONY: help clean html dirhtml singlehtml livehtml epub latex pdf

help:
	@echo "请使用以下命令构建文档:"
	@echo "  make clean       删除所有构建文件"
	@echo "  make html        生成标准HTML文档"
	@echo "  make livehtml    启动实时预览服务器（开发模式）"
	@echo "  make pdf         生成PDF文档（需要LaTeX）"
	@echo "  make check       验证文档完整性"
	@echo "  make coverage    生成API覆盖率报告"

clean:
	rm -rf $(BUILDDIR)/*
	@echo "已清除所有构建产物"

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo
	@echo "构建完成。打开 $(BUILDDIR)/html/index.html 查看文档"

livehtml:
	@sphinx-autobuild -b html \
		--port $(LIVEPORT) \
		--ignore $(IGNORE_PATTERNS) \
		"$(SOURCEDIR)" "$(BUILDDIR)/html" \
		$(SPHINXOPTS)
	@echo
	@echo "实时预览运行中：http://localhost:$(LIVEPORT)"

pdf:
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS)
	@echo "生成LaTeX文件..."
	@cd "$(BUILDDIR)/latex" && make
	@cp "$(BUILDDIR)/latex/*.pdf" "$(BUILDDIR)/pdf/"
	@echo "PDF文档已生成至 $(BUILDDIR)/pdf/"

check:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)
	@echo
	@echo "链接检查完成。查看报告：$(BUILDDIR)/linkcheck/output.txt"

coverage:
	$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS)
	@echo
	@echo "API覆盖率报告：$(BUILDDIR)/coverage/python.txt"

# 高级构建目标
apidoc:
	@sphinx-apidoc -o "$(SOURCEDIR)/api" "../src/dro" -f -e
	@echo "API文档模板已更新"

docker-build:
	@docker build -t dro-docs .
	@echo "文档Docker镜像构建完成"

docker-serve:
	@docker run -p $(LIVEPORT):$(LIVEPORT) -v $(PWD):/docs dro-docs make livehtml

# 开发助手
watch:
	@while true; do \
		inotifywait -r -e modify,move,create,delete "$(SOURCEDIR)"; \
		make html; \
	done
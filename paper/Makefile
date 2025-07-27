# QMANN Paper Makefile
# Provides commands for building, checking, and submitting the research paper

.PHONY: help pdf clean check-pdfa check-refs check-figs submit-arxiv

# Default target
help:
	@echo "QMANN Paper Build Commands"
	@echo "========================="
	@echo ""
	@echo "Build:"
	@echo "  pdf          Build main PDF"
	@echo "  pdf-fast     Quick build (no bibliography)"
	@echo "  clean        Clean build artifacts"
	@echo ""
	@echo "Quality Checks:"
	@echo "  check-pdfa   Check PDF/A compliance"
	@echo "  check-refs   Check references and citations"
	@echo "  check-figs   Check figure sizes and formats"
	@echo "  check-spell  Spell check LaTeX files"
	@echo "  check-all    Run all quality checks"
	@echo ""
	@echo "Submission:"
	@echo "  submit-arxiv Prepare arXiv submission package"
	@echo "  validate     Validate submission requirements"
	@echo ""
	@echo "Development:"
	@echo "  watch        Watch for changes and rebuild"
	@echo "  count        Count words and pages"
	@echo "  diff         Show changes since last commit"

# Build targets
pdf: main.pdf

main.pdf: main.tex refs.bib sections/*.tex
	@echo "Building PDF..."
	latexmk -pdf -file-line-error -halt-on-error -interaction=nonstopmode main.tex
	@echo "PDF build complete: main.pdf"

pdf-fast:
	@echo "Quick PDF build (no bibliography)..."
	pdflatex -file-line-error -halt-on-error -interaction=nonstopmode main.tex
	@echo "Quick build complete: main.pdf"

# Clean targets
clean:
	@echo "Cleaning build artifacts..."
	latexmk -C
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.toc *.nav *.snm *.vrb
	rm -f *.synctex.gz *.figlist *.makefile *.auxlock
	rm -f sections/*.aux
	rm -rf out/
	@echo "Clean complete"

clean-all: clean
	rm -f main.pdf
	rm -f *.pdf

# Quality check targets
check-pdfa: main.pdf
	@echo "Checking PDF/A compliance..."
	@if command -v qpdf >/dev/null 2>&1; then \
		echo "Validating PDF structure..."; \
		qpdf --check main.pdf && echo "✓ PDF structure valid"; \
	else \
		echo "⚠ qpdf not found, skipping PDF validation"; \
	fi
	
	@if command -v gs >/dev/null 2>&1; then \
		echo "Converting to PDF/A-1b..."; \
		gs -dPDFA=1 -dBATCH -dNOPAUSE -dUseCIEColor -sProcessColorModel=DeviceCMYK \
		   -sDEVICE=pdfwrite -sPDFACompatibilityPolicy=1 \
		   -sOutputFile=main_pdfa.pdf main.pdf; \
		echo "✓ PDF/A conversion complete: main_pdfa.pdf"; \
	else \
		echo "⚠ Ghostscript not found, skipping PDF/A conversion"; \
	fi
	
	@if command -v pdffonts >/dev/null 2>&1; then \
		echo "Checking font embedding..."; \
		pdffonts main.pdf | grep -v "emb" | grep -v "Type" | grep -v "---" || echo "✓ All fonts embedded"; \
	else \
		echo "⚠ pdffonts not found, skipping font check"; \
	fi
	
	@echo "Checking file size..."
	@SIZE=$$(stat -f%z main.pdf 2>/dev/null || stat -c%s main.pdf 2>/dev/null || echo "0"); \
	MAX_SIZE=$$((15 * 1024 * 1024)); \
	if [ $$SIZE -gt $$MAX_SIZE ]; then \
		echo "❌ PDF size ($$SIZE bytes) exceeds arXiv limit (15 MB)"; \
		exit 1; \
	else \
		echo "✓ PDF size OK: $$(echo "scale=1; $$SIZE / 1024 / 1024" | bc -l) MB"; \
	fi

check-refs:
	@echo "Checking references and citations..."
	@echo "Counting references..."
	@REF_COUNT=$$(grep -c "^@" refs.bib || echo "0"); \
	echo "Reference count: $$REF_COUNT"; \
	if [ $$REF_COUNT -gt 50 ]; then \
		echo "⚠ Many references ($$REF_COUNT > 50), consider reducing"; \
	else \
		echo "✓ Reference count OK"; \
	fi
	
	@echo "Checking for broken citations..."
	@BROKEN=0; \
	for cite in $$(grep -o '\\cite{[^}]*}' *.tex sections/*.tex | sed 's/.*\\cite{\([^}]*\)}.*/\1/' | sort -u); do \
		if ! grep -q "^@.*{$$cite," refs.bib; then \
			echo "❌ Broken citation: $$cite"; \
			BROKEN=$$((BROKEN + 1)); \
		fi; \
	done; \
	if [ $$BROKEN -eq 0 ]; then \
		echo "✓ All citations valid"; \
	else \
		echo "❌ Found $$BROKEN broken citations"; \
		exit 1; \
	fi

check-figs:
	@echo "Checking figures..."
	@if [ ! -d "figs" ]; then \
		echo "⚠ No figs directory found"; \
		exit 0; \
	fi
	
	@echo "Checking figure sizes..."
	@OVERSIZED=0; \
	for fig in figs/*; do \
		if [ -f "$$fig" ]; then \
			SIZE=$$(stat -f%z "$$fig" 2>/dev/null || stat -c%s "$$fig" 2>/dev/null || echo "0"); \
			SIZE_MB=$$(echo "scale=2; $$SIZE / 1024 / 1024" | bc -l); \
			if [ $$(echo "$$SIZE > 10485760" | bc -l) -eq 1 ]; then \
				echo "❌ Oversized figure: $$fig ($$SIZE_MB MB)"; \
				OVERSIZED=$$((OVERSIZED + 1)); \
			else \
				echo "✓ $$fig ($$SIZE_MB MB)"; \
			fi; \
		fi; \
	done; \
	if [ $$OVERSIZED -eq 0 ]; then \
		echo "✓ All figures within size limits"; \
	else \
		echo "❌ Found $$OVERSIZED oversized figures"; \
		exit 1; \
	fi
	
	@echo "Checking for missing figures..."
	@MISSING=0; \
	for fig in $$(grep -o '\\includegraphics[^{]*{[^}]*}' *.tex sections/*.tex | sed 's/.*{\([^}]*\)}.*/\1/'); do \
		if [ ! -f "$$fig" ] && [ ! -f "figs/$$fig" ]; then \
			echo "❌ Missing figure: $$fig"; \
			MISSING=$$((MISSING + 1)); \
		fi; \
	done; \
	if [ $$MISSING -eq 0 ]; then \
		echo "✓ All referenced figures found"; \
	else \
		echo "❌ Found $$MISSING missing figures"; \
		exit 1; \
	fi

check-spell:
	@echo "Checking spelling..."
	@if command -v aspell >/dev/null 2>&1; then \
		echo "Creating custom dictionary..."; \
		cat > .aspell.en.pws << 'EOF'; \
personal_ws-1.1 en 100 \
QRAM \
QMANN \
qubit \
qubits \
superposition \
entanglement \
Qiskit \
PennyLane \
arXiv \
Zenodo \
ORCID \
MNIST \
CIFAR \
LSTM \
NTM \
DNC \
VQE \
QAOA \
NISQ \
EOF \
		echo "Spell checking LaTeX files..."; \
		for file in *.tex sections/*.tex; do \
			if [ -f "$$file" ]; then \
				echo "Checking $$file..."; \
				aspell --mode=tex --personal=.aspell.en.pws list < "$$file" | sort -u > "$$file.misspelled"; \
				if [ -s "$$file.misspelled" ]; then \
					echo "⚠ Potential misspellings in $$file:"; \
					cat "$$file.misspelled"; \
				else \
					echo "✓ $$file"; \
				fi; \
				rm -f "$$file.misspelled"; \
			fi; \
		done; \
		rm -f .aspell.en.pws; \
	else \
		echo "⚠ aspell not found, skipping spell check"; \
	fi

check-all: check-pdfa check-refs check-figs check-spell
	@echo "✓ All quality checks completed"

# Submission targets
submit-arxiv: main.pdf check-all
	@echo "Preparing arXiv submission package..."
	mkdir -p out/arxiv_submission
	
	@echo "Copying necessary files..."
	cp main.tex out/arxiv_submission/
	cp refs.bib out/arxiv_submission/
	cp -r sections/ out/arxiv_submission/
	cp -r figs/ out/arxiv_submission/
	
	@echo "Copying style files..."
	@for style in *.cls *.sty; do \
		if [ -f "$$style" ]; then \
			cp "$$style" out/arxiv_submission/; \
		fi; \
	done
	
	@echo "Creating submission archive..."
	cd out/arxiv_submission && tar -czf ../arxiv_submission.tar.gz *
	
	@echo "Validating submission..."
	@cd out/arxiv_submission && \
	latexmk -pdf -file-line-error -halt-on-error -interaction=nonstopmode main.tex && \
	echo "✓ Submission package builds successfully"
	
	@echo "arXiv submission package ready: out/arxiv_submission.tar.gz"

validate: main.pdf
	@echo "Validating submission requirements..."
	
	@echo "Checking metadata..."
	@if [ ! -f "meta.yaml" ]; then \
		echo "❌ meta.yaml not found"; \
		exit 1; \
	else \
		echo "✓ meta.yaml found"; \
	fi
	
	@echo "Validating YAML syntax..."
	@if command -v python3 >/dev/null 2>&1; then \
		python3 -c "import yaml; yaml.safe_load(open('meta.yaml'))" && echo "✓ meta.yaml valid"; \
	else \
		echo "⚠ Python not found, skipping YAML validation"; \
	fi
	
	@echo "Checking required fields..."
	@python3 -c " \
import yaml; \
meta = yaml.safe_load(open('meta.yaml')); \
required = ['title', 'authors', 'abstract', 'categories']; \
missing = [f for f in required if f not in meta]; \
if missing: \
    print(f'❌ Missing required fields: {missing}'); \
    exit(1); \
else: \
    print('✓ All required metadata fields present'); \
"
	
	@echo "✓ Validation complete"

# Development targets
watch:
	@echo "Watching for changes... (Press Ctrl+C to stop)"
	@if command -v fswatch >/dev/null 2>&1; then \
		fswatch -o *.tex sections/*.tex refs.bib | while read; do \
			echo "Changes detected, rebuilding..."; \
			make pdf-fast; \
		done; \
	elif command -v inotifywait >/dev/null 2>&1; then \
		while inotifywait -e modify *.tex sections/*.tex refs.bib; do \
			echo "Changes detected, rebuilding..."; \
			make pdf-fast; \
		done; \
	else \
		echo "❌ No file watching utility found (fswatch or inotifywait)"; \
		exit 1; \
	fi

count:
	@echo "Document statistics:"
	@if command -v texcount >/dev/null 2>&1; then \
		texcount -total -brief main.tex; \
	else \
		echo "Word count (approximate):"; \
		wc -w *.tex sections/*.tex | tail -1; \
		echo "Page count:"; \
		if [ -f main.pdf ]; then \
			if command -v pdfinfo >/dev/null 2>&1; then \
				pdfinfo main.pdf | grep Pages; \
			else \
				echo "PDF exists but pdfinfo not available"; \
			fi; \
		else \
			echo "PDF not found, run 'make pdf' first"; \
		fi; \
	fi

diff:
	@echo "Changes since last commit:"
	@if git rev-parse --git-dir > /dev/null 2>&1; then \
		git diff --name-only HEAD~1 HEAD | grep -E '\.(tex|bib)$$' || echo "No LaTeX changes"; \
		echo ""; \
		echo "Detailed changes:"; \
		git diff HEAD~1 HEAD -- '*.tex' '*.bib'; \
	else \
		echo "Not a git repository"; \
	fi

# Archive target
archive:
	@echo "Creating project archive..."
	@DATE=$$(date +%Y%m%d); \
	tar -czf "qmann_paper_$$DATE.tar.gz" \
		--exclude="*.pdf" \
		--exclude="out/" \
		--exclude=".git/" \
		--exclude="*.aux" \
		--exclude="*.log" \
		*.tex *.bib sections/ figs/ Makefile meta.yaml checklist.md
	@echo "Archive created: qmann_paper_$$(date +%Y%m%d).tar.gz"

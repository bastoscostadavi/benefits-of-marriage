# Build the AAMAS submission. `make` produces marriage.pdf.
PAPER = marriage

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex bib.bib $(wildcard figures/*.pdf)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

# Refresh figures/ from results/ after re-running the simulations
figures: 
	./sync_figures.sh

# Anonymous zip for OpenReview: sources + figures, no build artefacts
zip: $(PAPER).pdf
	zip -r $(PAPER)-submission.zip $(PAPER).tex bib.bib aamas.cls \
	    ACM-Reference-Format.bst figures/ -x '*.DS_Store'

clean:
	rm -f $(PAPER).{aux,bbl,blg,log,out,synctex.gz} $(PAPER)-submission.zip

.PHONY: all figures zip clean

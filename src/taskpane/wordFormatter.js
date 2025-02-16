export class WordFormatter {
    constructor(context) {
        this.context = context;
    }

    async formatDocument() {
        const body = this.context.document.body;

        // Set basic document formatting
        body.font.name = "Times New Roman";
        body.font.size = 12;
        
        // Set margins (1 inch = 72 points)
        body.paragraphs.spacing = 24; // Double spacing
        const sections = body.sections;
        sections.load("items");
        await this.context.sync();

        sections.items.forEach(section => {
            section.pageSetup.topMargin = 72;
            section.pageSetup.bottomMargin = 72;
            section.pageSetup.leftMargin = 72;
            section.pageSetup.rightMargin = 72;
        });

        await this.context.sync();
    }

    async addTitle(title) {
        const range = this.context.document.body.getRange(Word.RangeLocation.start);
        range.insertParagraph(title, Word.InsertLocation.before);
        const titleParagraph = range.paragraphs.getFirst();
        
        titleParagraph.font.size = 16;
        titleParagraph.font.bold = true;
        titleParagraph.alignment = Word.Alignment.center;
        
        await this.context.sync();
    }

    async addPageNumbers() {
        const sections = this.context.document.sections;
        sections.load("items");
        await this.context.sync();

        sections.items[0].footers.primary.insertText("{PAGE}", Word.InsertLocation.replace);
        
        await this.context.sync();
    }

    async insertReferences(referencesHtml) {
        // Add page break before references
        const range = this.context.document.body.getRange(Word.RangeLocation.end);
        range.insertBreak(Word.BreakType.page);
        
        // Add References heading
        const heading = range.insertParagraph("References", Word.InsertLocation.after);
        heading.font.bold = true;
        heading.font.size = 16;
        
        // Insert formatted references
        const referencesRange = heading.getRange(Word.RangeLocation.after);
        referencesRange.insertHtml(referencesHtml, Word.InsertLocation.after);
        
        await this.context.sync();
    }

    async formatParagraphs() {
        const paragraphs = this.context.document.body.paragraphs;
        paragraphs.load("items");
        await this.context.sync();

        paragraphs.items.forEach(paragraph => {
            // Set first line indent
            paragraph.firstLineIndent = 36; // 0.5 inch = 36 points
            
            // Ensure consistent spacing
            paragraph.spacing.before = 0;
            paragraph.spacing.after = 0;
            paragraph.spacing.line = 24; // Double spacing
        });

        await this.context.sync();
    }

    async insertTableOfContents() {
        const range = this.context.document.body.getRange(Word.RangeLocation.start);
        range.insertBreak(Word.BreakType.page);
        
        const tocHeading = range.insertParagraph("Table of Contents", Word.InsertLocation.before);
        tocHeading.font.bold = true;
        tocHeading.font.size = 14;
        
        // Note: Actual TOC insertion requires field codes which aren't directly supported by Office JS
        // We'll need to implement a custom TOC generation based on the document's headings
        
        await this.context.sync();
    }
}

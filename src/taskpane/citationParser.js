import { Cite } from 'citation-js';

export class CitationParser {
    constructor() {
        this.citationMap = new Map();
        this.citationRegexes = {
            bibtex: /@[\w-]+\{[^}]+\}/g,
            inline: /\[(\d+)\]/g,
            doi: /doi:\s*(10\.\d{4,}\/[-._;()\/:A-Z0-9]+)/gi,
            url: /url:\s*(https?:\/\/[^\s]+)/gi,
            endnote: /(%[0-9A-Z]\s.+?(?=(%[0-9A-Z]|\Z)))/gs,
            ris: /(TY\s+-\s+.+?(?=TY\s+-|\Z))/gs,
            zotero: /\{\{([^}]+)\}\}/g
        };
    }

    async parseCitations(text, bibFile = null) {
        const citations = [];
        const promises = [];

        // If bibliography file is provided, parse it first
        if (bibFile) {
            try {
                const bibCitations = await this.parseBibliographyFile(bibFile);
                citations.push(...bibCitations);
            } catch (error) {
                console.error('Error parsing bibliography file:', error);
            }
        }

        // Find all citation formats in the text
        for (const [type, regex] of Object.entries(this.citationRegexes)) {
            let match;
            while ((match = regex.exec(text)) !== null) {
                promises.push(this.processCitation(match[0], type));
            }
        }

        // Wait for all citations to be processed
        const results = await Promise.all(promises);
        citations.push(...results.filter(c => c !== null));

        return this.deduplicateCitations(citations);
    }

    async parseBibliographyFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = async (event) => {
                try {
                    const content = event.target.result;
                    const fileExtension = file.name.split('.').pop().toLowerCase();
                    
                    let citations = [];
                    switch (fileExtension) {
                        case 'bib':
                            citations = await this.parseBibTeX(content);
                            break;
                        case 'enw':
                            citations = await this.parseEndNote(content);
                            break;
                        case 'ris':
                            citations = await this.parseRIS(content);
                            break;
                        default:
                            throw new Error('Unsupported bibliography file format');
                    }
                    resolve(citations);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Error reading file'));
            reader.readAsText(file);
        });
    }

    async parseBibTeX(content) {
        try {
            const citations = await Cite.async(content);
            return citations.data.map(citation => ({
                original: content,
                parsed: new Cite(citation),
                type: 'bibtex'
            }));
        } catch (error) {
            console.error('Error parsing BibTeX:', error);
            return [];
        }
    }

    async parseEndNote(content) {
        const citations = [];
        const matches = content.match(this.citationRegexes.endnote) || [];
        
        for (const match of matches) {
            try {
                const citation = await Cite.async(match, { forceType: '@endnote' });
                citations.push({
                    original: match,
                    parsed: citation,
                    type: 'endnote'
                });
            } catch (error) {
                console.warn('Failed to parse EndNote citation:', match);
            }
        }
        
        return citations;
    }

    async parseRIS(content) {
        const citations = [];
        const matches = content.match(this.citationRegexes.ris) || [];
        
        for (const match of matches) {
            try {
                const citation = await Cite.async(match, { forceType: '@ris' });
                citations.push({
                    original: match,
                    parsed: citation,
                    type: 'ris'
                });
            } catch (error) {
                console.warn('Failed to parse RIS citation:', match);
            }
        }
        
        return citations;
    }

    async processCitation(citation, type) {
        try {
            let citationData;
            switch (type) {
                case 'bibtex':
                    citationData = await Cite.async(citation);
                    break;
                case 'doi':
                    citationData = await Cite.async(citation.replace(/doi:\s*/, ''), {
                        type: 'doi'
                    });
                    break;
                case 'url':
                    citationData = await Cite.async(citation.replace(/url:\s*/, ''), {
                        type: 'url'
                    });
                    break;
                case 'zotero':
                    citationData = await this.processZoteroCitation(citation);
                    break;
                default:
                    return null;
            }

            return {
                original: citation,
                parsed: citationData,
                type: type
            };
        } catch (error) {
            console.warn(`Failed to parse citation: ${citation}`, error);
            return null;
        }
    }

    async processZoteroCitation(citation) {
        // Extract Zotero citation key
        const match = this.citationRegexes.zotero.exec(citation);
        if (!match) return null;

        const citationKey = match[1];
        // In a real implementation, this would interact with Zotero's API
        // For now, we'll create a basic citation
        return new Cite({
            id: citationKey,
            type: 'article',
            title: `Citation ${citationKey}`,
            author: [{ family: 'Unknown', given: 'Author' }],
            issued: { 'date-parts': [[new Date().getFullYear()]] }
        });
    }

    formatCitation(citation, style) {
        try {
            return citation.parsed.format('bibliography', {
                format: 'html',
                template: this.mapStyle(style),
                lang: 'en-US'
            });
        } catch (error) {
            console.warn(`Failed to format citation: ${citation.original}`, error);
            return citation.original;
        }
    }

    mapStyle(style) {
        // Map our UI styles to citation.js templates
        const styleMap = {
            'apa': 'apa',
            'harvard': 'harvard1',
            'mla': 'modern-language-association',
            'chicago': 'chicago-author-date',
            'ieee': 'ieee',
            'acm': 'acm'
        };
        return styleMap[style] || 'apa';
    }

    deduplicateCitations(citations) {
        const seen = new Set();
        return citations.filter(citation => {
            const key = citation.parsed.format('bibtex');
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    }
}

export const paperFormats = {
    default: {
        name: 'Default Academic',
        margins: { top: 72, bottom: 72, left: 72, right: 72 }, // 1 inch = 72 points
        font: { name: 'Times New Roman', size: 12 },
        spacing: 24, // Double spacing
        headingStyles: {
            1: { size: 16, bold: true, align: 'center' },
            2: { size: 14, bold: true, align: 'left' },
            3: { size: 12, bold: true, align: 'left' },
            4: { size: 12, bold: true, italic: true, align: 'left' }
        }
    },
    ieee: {
        name: 'IEEE Conference',
        margins: { top: 72, bottom: 72, left: 72, right: 72 },
        font: { name: 'Times New Roman', size: 10 },
        spacing: 14.4, // 1.2 spacing
        headingStyles: {
            1: { size: 14, bold: true, align: 'center', case: 'uppercase' },
            2: { size: 12, bold: true, align: 'left', case: 'capitalize' },
            3: { size: 10, bold: true, align: 'left', case: 'sentence' },
            4: { size: 10, bold: true, italic: true, align: 'left' }
        },
        citationStyle: 'ieee',
        specialFeatures: {
            abstractStyle: 'italic',
            keywordsRequired: true,
            columnCount: 2
        }
    },
    acm: {
        name: 'ACM Conference',
        margins: { top: 72, bottom: 72, left: 72, right: 72 },
        font: { name: 'Times New Roman', size: 10 },
        spacing: 12, // Single spacing
        headingStyles: {
            1: { size: 14, bold: true, align: 'center' },
            2: { size: 12, bold: true, align: 'left' },
            3: { size: 10, bold: true, align: 'left' },
            4: { size: 10, bold: true, italic: true, align: 'left' }
        },
        citationStyle: 'acm',
        specialFeatures: {
            abstractStyle: 'regular',
            keywordsRequired: true,
            categoriesRequired: true
        }
    },
    apa: {
        name: 'APA 7th Edition',
        margins: { top: 72, bottom: 72, left: 72, right: 72 },
        font: { name: 'Times New Roman', size: 12 },
        spacing: 24, // Double spacing
        headingStyles: {
            1: { size: 14, bold: true, align: 'center' },
            2: { size: 14, bold: true, align: 'left' },
            3: { size: 12, bold: true, align: 'left', indent: 36 },
            4: { size: 12, bold: true, align: 'left', indent: 36 }
        },
        citationStyle: 'apa',
        specialFeatures: {
            runningHead: true,
            pageNumberLocation: 'topRight'
        }
    }
};

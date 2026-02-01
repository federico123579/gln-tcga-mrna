// ============================================================================
// GLN-TCGA Report Template - Politecnico di Milano Style
// ============================================================================

// ============================================================================
// SYMBOL SHORTCUTS - Consistent math notation
// ============================================================================

// Dataset parameters
#let nsamples = $N_"samples"$
#let ngenes = $N_"genes"$
#let ntumor = $N_"tumor"$
#let nnormal = $N_"normal"$

// Model parameters
#let ctxdim = $K$
#let nlayers = $L$
#let ctxV = $bold(V)_"ctx"$
#let ctxb = $bold(b)_"ctx"$
#let Wmat = $bold(W)$

// Metrics
#let acc = $"Acc"$
#let IGscore = $"IG"$

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Assessment table with color-coded status
/// Usage: #assessment-table((("Aspect", "good"), ("Other", "bad")))
#let assessment-table(items) = {
  figure(
    table(
      columns: (auto, auto),
      inset: 8pt,
      align: (left + horizon, center + horizon),
      stroke: 0.5pt,
      table.header([*Aspect*], [*Assessment*]),
      table.hline(),
      ..items
        .map(((aspect, assessment)) => (
          aspect,
          if assessment == "good" {
            text(fill: green.darken(20%), sym.checkmark + " Excellent")
          } else if assessment == "bad" {
            text(fill: red.darken(20%), sym.times + " Not suitable")
          } else if assessment == "warn" {
            text(fill: orange.darken(10%), sym.diamond.stroked + " Limited")
          } else { assessment },
        ))
        .flatten(),
    ),
  )
}

/// Gene ranking table
#let gene-table(genes, caption: none) = {
  figure(
    table(
      columns: (auto, auto, auto, auto),
      inset: 6pt,
      align: (left, center, center, left),
      table.header([*Gene*], [*IG Rank*], [*Score*], [*Role*]),
      table.hline(),
      ..genes.map(g => (g.name, str(g.rank), str(g.score), g.role)).flatten(),
    ),
    caption: caption,
  )
}

/// Code block with caption
#let code-figure(code-content, caption: none) = {
  figure(
    block(
      fill: luma(245),
      inset: 10pt,
      radius: 4pt,
      width: 100%,
      code-content,
    ),
    caption: caption,
    supplement: "Listing",
    kind: "listing",
  )
}

/// Side-by-side comparison grid
#let side-by-side(left-content, right-content, left-caption: none, right-caption: none) = {
  grid(
    columns: (1fr, 1fr),
    gutter: 16pt,
    figure(left-content, caption: left-caption), figure(right-content, caption: right-caption),
  )
}

/// Parameter table for experiments
#let param-table(title, ..params) = {
  let pairs = params.pos()
  figure(
    table(
      columns: pairs.len(),
      inset: 5pt,
      align: center + horizon,
      ..pairs.map(p => p.at(0)),
      table.hline(),
      ..pairs.map(p => {
        let v = p.at(1)
        if type(v) == str { [_#v _] } else { [#v] }
      })
    ),
    caption: [*#title*],
    supplement: none,
  )
}

/// Grouped experiment parameters (dataset, model, training)
#let experiment-params(
  dataset: (),
  model: (),
  training: (),
) = {
  set table(inset: 5pt)
  show table.cell.where(y: 0): strong
  align(center, grid(
    columns: (auto, auto, auto),
    gutter: 10pt,
    if dataset.len() > 0 { param-table("Dataset", ..dataset) },
    if model.len() > 0 { param-table("Model", ..model) },
    if training.len() > 0 { param-table("Training", ..training) },
  ))
}

/// Note/callout box
#let note-box(content, title: "Note", color: blue) = {
  block(
    fill: color.lighten(90%),
    stroke: (left: 3pt + color),
    inset: 10pt,
    radius: (right: 4pt),
    width: 100%,
    [*#title:* #content],
  )
}

#let warning-box(content) = note-box(content, title: "Warning", color: orange)
#let finding-box(content) = note-box(content, title: "Key Finding", color: green.darken(20%))
#let critical-box(content) = note-box(content, title: "Critical", color: red)

/// Mathematical definition
#let definition(term, body) = {
  block(
    inset: (left: 1em, y: 0.5em),
    [*#term*: #body],
  )
}

/// Results comparison matrix with auto-highlighting
#let results-matrix(headers, ..rows) = {
  let row-data = rows.pos()
  table(
    columns: headers.len(),
    inset: 6pt,
    align: center + horizon,
    table.header(..headers.map(h => [*#h*])),
    table.hline(),
    ..row-data
      .map(row => {
        row
          .enumerate()
          .map(((i, cell)) => {
            if i > 0 and cell == "Satisfied" {
              text(fill: green.darken(20%), cell)
            } else if i > 0 and cell == "Unsatisfied" {
              text(fill: red.darken(20%), cell)
            } else {
              cell
            }
          })
      })
      .flatten()
  )
}

/// Figure placeholder for TODO figures
#let figure-placeholder(width: 100%, height: 5cm, caption: none) = {
  figure(
    block(
      width: width,
      height: height,
      fill: luma(240),
      stroke: 1pt + luma(200),
      align(center + horizon, text(fill: luma(120), size: 11pt, [TODO: Add figure])),
    ),
    caption: caption,
  )
}

/// Comparison table (two columns: before/after, sparse/expressed, etc.)
#let comparison-table(left-title, right-title, left-items, right-items) = {
  grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    [
      *#left-title*
      #table(
        columns: 2,
        inset: 5pt,
        ..left-items.flatten()
      )
    ],
    [
      *#right-title*
      #table(
        columns: 2,
        inset: 5pt,
        ..right-items.flatten()
      )
    ],
  )
}

// ============================================================================
// TEMPLATE FUNCTIONS
// ============================================================================

/// Author formatting for title page
#let author(
  fullname,
  id_number,
  person_code,
) = [
  #[
    #set text(size: 12pt)
    #let mail = str(person_code) + "@polimi.it"
    #fullname #link("mailto:" + mail)[`<`#raw(mail)`>`]
  ]
]

/// Cover page layout
#let template_cover(
  title,
  subtitle,
  course_name,
  abstract: "",
  instructors: (),
  authors: (),
  academic_year: "2025-2026",
) = (
  [
    #set align(center)
    #set text(size: 11pt)

    #image("assets/logo_politecnico.png", width: 45%)
    #v(20pt)
    #block(width: 80%, text(22pt, smallcaps(course_name)))
    #v(-10pt)
    #text(24pt)[*#title*]
    #v(-10pt)
    #text(17pt)[#subtitle]
    #v(5pt)

    #if instructors.len() > 0 [
      #align(center, text(15pt)[Instructors:])
      #v(-10pt)
      #instructors.join(",\n")
      #v(5pt)
    ]

    #align(center, text(15pt)[Authors:])
    #v(-10pt)
    #authors.join(",\n")
    #v(5pt)
    #text(13pt)[A.Y. #academic_year]
    #v(5pt)
    #align(center)[
      #smallcaps("Abstract")
      #block(width: 82%, par(justify: true, text(12pt, abstract)))
    ]
  ]
)

/// Table of contents
#let index(
  title: "Contents",
) = outline(
  depth: 2,
  title: title,
  indent: auto,
)

/// Main template wrapper
#let template(
  authors: (),
  instructors: (),
  title: "",
  subtitle: "",
  course_name: "",
  meta_title: "",
  abstract: "",
  academic_year: "2025-2026",
  doc,
) = [
  #let cover_authors = ()
  #let authors_full_names = ()
  #for (fullname, id_number, person_code) in authors {
    cover_authors.push(author(
      fullname,
      id_number,
      person_code,
    ))
    authors_full_names.push(fullname)
  }

  #set document(title: if meta_title != "" { meta_title } else { title }, author: authors_full_names)
  #show link: set text(fill: maroon)

  #let template_authors = ()
  #for (name, id, code) in authors {
    template_authors.push(author(name, id, code))
  }
  #let template_instructors = ()
  #for (name) in instructors {
    template_instructors.push(text(size: 15pt, smallcaps(name)))
  }

  #template_cover(
    title,
    subtitle,
    course_name,
    instructors: template_instructors,
    authors: template_authors,
    abstract: abstract,
    academic_year: academic_year,
  )
  #pagebreak()

  #index()
  #pagebreak()

  #set par(justify: true)
  #set text(size: 10pt)
  #set heading(numbering: "1.")
  #set page(
    margin: (x: 2cm, y: 2cm),
    footer: context [
      #set align(center)
      #set text(9pt)
      #counter(page).display("1")
    ],
  )

  // Show rules for styling
  #show link: underline
  #show ref: underline

  #doc

  // Bibliography (uncomment when bibliography.yml is ready)
  #pagebreak()
  = References <references>
  #bibliography("bibliography.yml", title: none)
]

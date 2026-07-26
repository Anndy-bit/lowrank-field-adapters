\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{scpe}[2026/05/13 v27.0 SCPE Ultimate Perfect Font Template]

% ==========================================
% Base Class Setup
% ==========================================
\LoadClass[11pt, a4paper, onecolumn]{article}

% ==========================================
% 1. Encoding and Fonts
% ==========================================
\RequirePackage[utf8]{inputenc}  
\RequirePackage[T1]{fontenc}     
\RequirePackage{mathpazo}
\RequirePackage{helvet}

% ==========================================
% 2. Page Layout
% ==========================================
\RequirePackage{geometry}
\geometry{a4paper, left=2.5cm, right=2.5cm, top=2.8cm, bottom=3.5cm, headheight=35pt, headsep=15pt}

% ==========================================
% 3. Brand Colors 
% ==========================================
\RequirePackage{xcolor}
\definecolor{SCPEblue}{RGB}{12, 54, 112}   
\definecolor{SCPEgray}{RGB}{100, 100, 100} 
\definecolor{orcidcolor}{HTML}{A6CE39} 

% ==========================================
% 4. Math, Graphics, Tables, and Advanced Boxes
% ==========================================
\RequirePackage{amsmath, amsfonts, amssymb} 
\RequirePackage{graphicx}                   
\RequirePackage{booktabs}                   
\RequirePackage{longtable}                  
\RequirePackage{multirow} 
\RequirePackage{caption}
\RequirePackage{environ} 
\RequirePackage{tcolorbox} 
\RequirePackage{tikz} 
\tcbuselibrary{skins, breakable}

\captionsetup{labelfont={bf, color=SCPEblue, sf}, textfont={it, sf}} 

\newcommand{\orcid}[1]{\href{https://orcid.org/#1}{\hspace{2pt}\textcolor{orcidcolor}{\tikz[baseline=-0.4ex]{\draw[fill=orcidcolor,draw=none] (0,0) circle (0.18); \node[text=white, font=\sffamily\bfseries, scale=0.55, inner sep=0pt] at (0,0) {iD};}}}}

% ==========================================
% ==========================================
\def\@articletype{Journal Article}
\newcommand{\articletype}[1]{\gdef\@articletype{#1}}

\def\@pubyear{2026}
\newcommand{\pubyear}[1]{\gdef\@pubyear{#1}}

\def\@volume{1}
\newcommand{\volume}[1]{\gdef\@volume{#1}}

\def\@issue{1}
\newcommand{\issue}[1]{\gdef\@issue{#1}}

\def\@doinum{10.1234/scpe.2026.001}
\newcommand{\doinum}[1]{\gdef\@doinum{#1}}

\def\@papercitation{Author. Title. \textit{Scalable Comput. Pract. Exp.} \textbf{2026}, \textit{1}(1).}
\newcommand{\papercitation}[1]{\gdef\@papercitation{#1}}

\def\@paperhistory{Received: -- \\ Accepted: -- \\ Published: --}
\newcommand{\paperhistory}[1]{\gdef\@paperhistory{#1}}

% ==========================================
% ==========================================
\RequirePackage[colorlinks=true, linkcolor=SCPEblue, citecolor=SCPEblue, urlcolor=SCPEblue]{hyperref}
\urlstyle{same} 

% ==========================================
% ==========================================
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{} 

\fancyhead[L]{\color{SCPEgray}\sffamily\footnotesize \textit{Scalable Computing: Practice and Experience} \textbf{\@pubyear}, \textit{\@volume}(\@issue)}
\fancyhead[R]{\color{SCPEgray}\sffamily\footnotesize \url{https://www.scpe.org}}
\fancyfoot[L]{\color{SCPEgray}\sffamily\footnotesize \copyright~\@pubyear\ SCPE.} 
\fancyfoot[C]{\color{SCPEblue}\sffamily\bfseries \thepage}                 
\fancyfoot[R]{}                                                             
\renewcommand{\headrulewidth}{0.5pt} 
\renewcommand{\headrule}{\hbox to\headwidth{\color{SCPEblue}\leaders\hrule height \headrulewidth\hfill}} 
\renewcommand{\footrulewidth}{0.5pt} 
\renewcommand{\footrule}{\hbox to\headwidth{\color{SCPEgray}\leaders\hrule height \footrulewidth\hfill}} 

\fancypagestyle{plain}{
  \fancyhf{}
  \fancyhead[L]{\color{SCPEblue}\sffamily\Large \textbf{SCPE} \color{SCPEgray}\large | \textit{\@articletype}}
  \fancyhead[R]{
      \begin{tabular}[b]{@{}r@{}}
          \color{SCPEgray}\sffamily\footnotesize \textit{Scalable Computing: Practice and Experience} \\
          \color{SCPEgray}\sffamily\footnotesize Volume \@volume, Issue \@issue, \@pubyear \\
          \color{SCPEblue}\bfseries OPEN ACCESS
      \end{tabular}
  }
  \fancyfoot[L]{\color{SCPEgray}\sffamily\footnotesize \copyright~\@pubyear\ SCPE.} 
  \fancyfoot[C]{\color{SCPEblue}\sffamily\bfseries \thepage}                 
  \fancyfoot[R]{}  
  \renewcommand{\headrulewidth}{1.5pt}
  \renewcommand{\footrulewidth}{0.5pt}
}

% ==========================================
% ==========================================
\newcommand{\papertitle}[1]{
    \thispagestyle{plain} 
    \vspace*{0pt} 
    {\noindent\raggedright\LARGE\sffamily\bfseries\color{SCPEblue} #1\par}
    \vspace{8pt} 
}

\newcommand{\paperauthor}[1]{
    {\noindent\raggedright\large\sffamily\bfseries #1\par}
    \vspace{4pt} 
}

\newcommand{\affiliation}[1]{
    {\noindent\raggedright\small\sffamily\color{SCPEgray}\itshape #1\par}
    \vspace{15pt} 
}

% ==========================================
% ==========================================
\def\@scpeabstract{}
\RenewEnviron{abstract}{\global\let\@scpeabstract\BODY}

\newcommand{\keywords}[1]{%
    \vspace{0pt} 
    \noindent
    \begin{minipage}[t]{0.30\textwidth}
        \vspace{0pt} 
        \begin{tcolorbox}[
            enhanced, nobeforeafter, colback=SCPEgray!5, colframe=SCPEgray!30, 
            arc=3mm, boxrule=0.5pt, left=4mm, right=4mm, top=4mm, bottom=4mm
        ]
        \sffamily\color{SCPEgray}\raggedright 
        
        {\fontsize{9.5pt}{12pt}\selectfont\textbf{\color{SCPEblue}How to cite:}}\\[3pt]
        {\fontsize{8pt}{11pt}\selectfont \@papercitation} \\[8pt]
        
        {\fontsize{9.5pt}{12pt}\selectfont\textbf{\color{SCPEblue}DOI:}}\\[3pt]
        {\fontsize{8pt}{11pt}\selectfont \href{https://doi.org/\@doinum}{\color{SCPEblue}\@doinum}} \\[4pt]
        \href{https://crossmark.crossref.org/dialog/?doi=\@doinum\&domain=pdf}{%
            \IfFileExists{crossmark.png}{\includegraphics[height=12pt]{crossmark.png}}{\framebox{\tiny Check for updates}}%
        }\\[8pt]
        
        {\fontsize{9.5pt}{12pt}\selectfont\textbf{\color{SCPEblue}Article History:}}\\[3pt]
        {\fontsize{8pt}{11pt}\selectfont \@paperhistory} \\[12pt] 
        
        \IfFileExists{ccby.png}{\includegraphics[width=0.55\linewidth]{ccby.png}}{\framebox[\linewidth]{\rule{0pt}{15pt}\sffamily CC-BY}}\\[6pt]
        {\fontsize{7.5pt}{10pt}\selectfont 
        \copyright~\@pubyear\ by the authors. Licensee SCPE.\\[2pt]
        This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license.}
        \end{tcolorbox}
    \end{minipage}%
    \hfill
    \begin{minipage}[t]{0.66\textwidth}
        \vspace{0pt} 
        \noindent\textcolor{SCPEblue}{\rule{\linewidth}{1pt}}\\[0.2cm]
        \noindent\sffamily\textbf{\color{SCPEblue} Abstract:}\quad\rmfamily\upshape \@scpeabstract
        \vspace{0.4cm}\par
        \noindent\sffamily\textbf{\color{SCPEblue} Keywords:}\quad\rmfamily\upshape #1
        \vspace{0.15cm}\par
        \noindent\textcolor{SCPEblue}{\rule{\linewidth}{1pt}}
    \end{minipage}
    \par\vspace{0.3cm}
}

% ==========================================
% ==========================================
\RequirePackage{titlesec}
\titleformat{\section}{\Large\sffamily\bfseries\color{SCPEblue}}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\large\sffamily\bfseries\color{SCPEblue}}{\thesubsection}{0.5em}{}
\titleformat{\subsubsection}{\normalsize\sffamily\bfseries\color{SCPEblue}}{\thesubsubsection}{0.5em}{}

% ==========================================
% ==========================================
\AtEndDocument{
    \vspace{0.8cm} 
    \begin{tcolorbox}[
        enhanced, breakable, colback=SCPEgray!5, colframe=SCPEgray!30, 
        arc=3mm, boxrule=0.5pt, left=6mm, right=6mm, top=4mm, bottom=4mm
    ]
    \sffamily\fontsize{8.5pt}{11pt}\selectfont\color{SCPEgray}
    \textbf{\color{SCPEblue}Disclaimer/Publisher's Note:} The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of SCPE and/or the editor(s). SCPE and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.
    \end{tcolorbox}
}

\endinput
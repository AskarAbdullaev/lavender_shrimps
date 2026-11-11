# AI reflection, Phase I


| First name | Last name | Student ID |
|---|---|---|
| Bernhard | Ruhm | K12019841 |

## How AI was used

**Claude Code** was used for most of the work. That includes code generation for running inference, calculating metrics and statistics, brainstorm ideas and preliminiary visualization drafts. 

However Claude Codes visual style choices were plain. **Claude Design** was used to restyled the page once, to get the final version. Interestignly, using Claude Design  broke the  UMAP input selection and had to be fixed afterwords.
## Strengths

- **Interactive plots became easy.** With little to no  UI/UX experience implementation of interactive plots with rich features, such as an image viewer with zoom and a lasso selction in the UMAP would take days. 
- **Broader analysis.** Very effective to work on a domain specific taks and model with no prior domain knowledge, as I have not worked with Cellpose model or celluar microscopy images before. Brainstorming on ideas with AI safed a lot of time.
## Getting it right is iterative

You are able to obtain a functional product very easily and fast, but getting one that satisfied the requirements regarding information and visual appeal took many iterations. The first versions were very "bloated" and information dense, as the plots contained a lot of additional statistics and explanation. Furthermore the interaction with the visualization were also quite poor and intuitive. Solving this required specfic requests and discussion about design choices. This could aso be because the first requests were probably too broad. 
## How AI suggestions affected the analysis

| Suggestion | Decision | Why |
|---|---|---|
| Rerun inference with Glioma dataset on different cell scales| Accepted | Quickly noticed that default cell size for Glioma was too small.|
| Introducing threshold sweep to find the best setting model setting cells| Rejected | Our goal is understanding the model, not tuning it to get the best prediction accuracy|
| Introducing Post-Processing that identifies if errors are fixable | Rejected | Again, goal is only understanding the model.|
| Replace the initial flow char with group bars | Accepted | The flow chart looked the same for every input; the bars revealed the input effects |
| Replacing image UMAP to cell UMAP | Accepted | Gave more insights and looked better|

## Failure patterns

1. Especially at the beginning, some responses suggested ways to improve segmentation quality, by changing threshold values and introducting post processing layer without beeing asked to do so
2. Some suggested plots looked nice but provided no real information. Before the bar diagram showing the influence of different input features on the output, a sankey diagram was used. But value changes were barely visable in the plot
## Best practices

- Having a clear idea of what you actually want to visualize beginning design process
- Be in control of analytical decissions
- Start small and add gradually to avoid bloated plots

- Seperate modelling of logic and visuals
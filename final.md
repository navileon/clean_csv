# Complete Multi-Method Analysis: Large vs Small Tech Companies

# Warning!!! 
This report has langrage re-written by LLM, may not be fullt accurate

**Corpus:** 11,929 paragraphs | 645,873 words  
**Methods:** Keywords, Sentiment, Readability, Topic Modeling (LDA)

---

## The Complete Picture

This document integrates **four complementary analytical methods** to provide a comprehensive understanding of how large and small tech companies communicate differently.

---

##  Method 1: Keyword Analysis (Statistical)

**What it reveals:** *Which specific terms distinguish each group*

### Large Companies (AWS, Microsoft)
**Top Distinctive Keywords (p < 0.001):**
- `aws` (LL=2003), `microsoft` (LL=1801), `amazon` (LL=1072)
- `customers` (LL=419), `solutions` (LL=566), `enterprise` (LL=333)
- `cloud` (LL=471), `security` (LL=80), `business` (LL=327)

**Language Pattern:** Corporate, solution-oriented, customer-centric

### Small Companies (Fly, Perplexity, Supabase)
**Top Distinctive Keywords (p < 0.001):**
- `supabase` (LL=1844), `perplexity` (LL=1094), `fly` (LL=947)
- `database` (LL=753), `postgres` (LL=773), `code` (LL=183)
- `app` (LL=401), `want` (LL=317), `like` (LL=405)

**Language Pattern:** Technical, conversational, developer-oriented

### Integration: Keywords → Topics
- Large keywords map to **"Cloud Infrastructure"** topic (38.9%)
- Small keywords map to **"Database & Storage"** topics (27.7%)

---

##  Method 2: Sentiment Analysis (Emotional Tone)

**What it reveals:** *The emotional coloring of communications*

| Metric | Large | Small | Difference |
|--------|-------|-------|------------|
| **VADER Compound** | +0.427 | +0.170 | +152% more positive |
| **Polarity** | +0.083 | +0.112 | Small slightly more positive |
| **Subjectivity** | 0.346 | 0.441 | Small +27% more subjective |

### Key Finding
**Large companies use more enthusiastic, optimistic language** despite more formal tone.

### Integration: Sentiment → Topics
- **Large positive sentiment** driven by Topic 1 (38.9%): "customers", "solutions", "success"
- **Small neutral tone** reflects technical focus: Topics 1+3+4 (47.3%): "database", "code", "functions"

---

##  Method 3: Readability & Complexity

**What it reveals:** *How accessible the writing is*

| Metric | Large | Small | Winner |
|--------|-------|-------|--------|
| **Grade Level** | 13.6 (college) | 7.9 (8th grade) | Small (simpler) |
| **Reading Ease** | 32.9 (difficult) | 63.1 (standard) | Small (+92%) |
| **Sentence Length** | 18.5 words | 13.0 words | Small (shorter) |
| **Vocabulary Diversity (TTR)** | 3.86% | 4.65% | Small (+20%) |

### Key Finding
**Small companies write at 8th-grade level, large companies require college education.**

### Integration: Complexity → Topics
- **Large complexity** from Topic 1 (38.9%): business narratives, formal solutions language
- **Small simplicity** from Topic 2 (15.7%): "create", "app", "start", tutorial-style

---

##  Method 4: Topic Modeling (LDA) - Content Strategy

**What it reveals:** *The abstract themes and story types each group tells*

### Large Companies: 8 Core Story Types

| Rank | Topic | Prevalence | Story Type |
|------|-------|------------|------------|
| 1 | Cloud Infrastructure | **38.9%** | "We enable business success" |
| 2 | Security & Compliance | 16.7% | "Enterprise-grade security" |
| 3 | Developer Tools & APIs | 9.5% | "Build on our platform" |
| 4 | IAM & Access Control | 7.9% | "Managing access at scale" |
| 5 | Infrastructure & Storage | 7.4% | "Global infrastructure" |
| 6 | AI/ML Platform | 7.3% | "AI at enterprise scale" |
| 7 | User Experience & Cost | 6.8% | "Optimizing value" |
| 8 | AI Agents & Products | 5.6% | "AI-powered products" |

**Dominant Narrative:** Business transformation (38.9%)

### Small Companies: 8 Core Story Types

| Rank | Topic | Prevalence | Story Type |
|------|-------|------------|------------|
| 1 | AI & Machine Learning | **19.9%** | "Practical ML implementations" |
| 2 | Developer Tools & APIs | 15.7% | "Build your next project" |
| 3 | Database Basics | 14.4% | "Database fundamentals" |
| 4 | Advanced Database | 13.3% | "Advanced techniques" |
| 5 | Functions & Queries | 13.0% | "Writing effective code" |
| 6 | Performance & ML | 7.6% | "Optimizing performance" |
| 7 | Search & Information | 5.9% | "Search & discovery" |
| 8 | Infrastructure Ops | 5.2% | "Infrastructure management" |

**Dominant Narrative:** Technical implementation (60%+ database/ML content)

---

##  Cross-Method Synthesis

### Finding 1: The 38.9% Explanation
**Why large companies are more complex and positive:**

- **Topic Modeling:** 38.9% is business success narratives
- **Keywords:** "customers", "solutions", "success" dominate
- **Sentiment:** Optimistic business transformation stories (+0.427)
- **Complexity:** Formal enterprise language (Grade 13.6)

→ **Insight:** Nearly 4 in 10 large company paragraphs tell customer success stories.

### Finding 2: The 60% Technical Focus
**Why small companies are simpler and more diverse:**

- **Topic Modeling:** 60% is database + ML technical content
- **Keywords:** "postgres", "database", "code", "functions"
- **Sentiment:** Neutral, fact-based documentation (+0.170)
- **Complexity:** Tutorial-style, 8th-grade reading level

→ **Insight:** 6 in 10 small company paragraphs are hands-on technical guides.

### Finding 3: Vocabulary Paradox
**Small companies: simpler words, MORE diversity**

- **Readability:** Small uses shorter words (4.87 vs 5.52 chars)
- **Diversity:** Small has 20% more vocabulary variety (TTR 4.65% vs 3.86%)
- **Topic Modeling:** Small covers more diverse topics (8 balanced vs 1 dominant)
- **Keywords:** Small shows varied technical vocabulary

→ **Insight:** Simple language ≠ repetitive. Technical variety creates lexical diversity.

### Finding 4: The Security Gap
**Large emphasizes security 4× more than small:**

- **Topic Modeling:** Large has 24.6% security topics, small has 5.9%
- **Keywords:** "security" has LL=80 for large (significant)
- **Sentiment:** Security topics contribute to positive trust messaging
- **Readability:** Security content is more complex (compliance jargon)

→ **Insight:** Enterprise customers demand extensive security narratives.

---



##  The Four-Method Visualization

```
                    LARGE COMPANIES (AWS, Microsoft)
                    
Method 1: Keywords  → customers, solutions, cloud, business, security
                    
Method 2: Sentiment → +0.427 (Highly Positive) | 0.346 (Objective)
                    
Method 3: Reading   → Grade 13.6 (College) | 18.5 words/sentence | 3.86% TTR
                    
Method 4: Topics    → Cloud Infrastructure (38.9%) + Security (24.6%)
                    
STRATEGY: "We enable enterprise business transformation"

═══════════════════════════════════════════════════════════════════════

                    SMALL COMPANIES (Fly, Perplexity, Supabase)
                    
Method 1: Keywords  → database, postgres, code, app, functions, query
                    
Method 2: Sentiment → +0.170 (Neutral) | 0.441 (Subjective)
                    
Method 3: Reading   → Grade 7.9 (8th grade) | 13.0 words/sentence | 4.65% TTR
                    
Method 4: Topics    → Database (27.7%) + ML (19.9%) + Dev Tools (15.7%)
                    
STRATEGY: "We teach developers to build better software"
```

---


##  Key Metrics Summary

| Metric | Large | Small | Winner | Method |
|--------|-------|-------|--------|--------|
| Sentiment (VADER) | +0.427 | +0.170 | Large (+152%) | Method 2 |
| Reading Level | Grade 13.6 | Grade 7.9 | Small (simpler) | Method 3 |
| Vocabulary Diversity | 3.86% | 4.65% | Small (+20%) | Method 3 |
| Business Content | 38.9% | 0% | Large | Method 4 |
| Technical Content | 31.7% | 60.0% | Small (+89%) | Method 4 |
| Security Content | 24.6% | 5.9% | Large (4×) | Method 4 |
| Developer Tutorials | 9.5% | 15.7% | Small (+65%) | Method 4 |

---

##  Future Research Directions

**Topic Correlation:** Which topics appear together (co-occurrence)
---

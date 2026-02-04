// Mock handlers for MSW (Mock Service Worker)
// These mock API responses for testing

import { http, HttpResponse } from 'msw'

const API_URL = 'http://localhost:8000/api'

export const handlers = [
  // Auth endpoints
  http.post(`${API_URL}/auth/register`, async ({ request }: { request: Request }) => {
    const body = await request.json() as { email: string; password: string; name: string }
    return HttpResponse.json({
      id: 'test-user-123',
      email: body.email,
      name: body.name,
    })
  }),

  http.post(`${API_URL}/auth/token`, async () => {
    return HttpResponse.json({
      access_token: 'mock-jwt-token',
      token_type: 'bearer',
    })
  }),

  http.get(`${API_URL}/auth/users/me`, async () => {
    return HttpResponse.json({
      id: 'test-user-123',
      email: 'test@example.com',
      name: 'Test User',
    })
  }),

  // Projects endpoints
  http.get(`${API_URL}/projects`, async () => {
    return HttpResponse.json([
      {
        id: 'project-1',
        title: 'AI in Education',
        research_question: 'How does AI affect learning?',
        keywords: ['AI', 'education'],
        subtopics: ['Adaptive Learning'],
        status: 'completed',
        total_papers_found: 25,
        created_at: '2024-01-15T10:00:00Z',
        agent_plans: [],
        paper_references: [],
      },
      {
        id: 'project-2',
        title: 'Climate Change Research',
        research_question: 'What are the impacts of climate change?',
        keywords: ['climate', 'environment'],
        subtopics: ['Rising Temperatures'],
        status: 'searching',
        total_papers_found: 0,
        created_at: '2024-01-20T14:30:00Z',
        agent_plans: [],
        paper_references: [],
      },
    ])
  }),

  http.get(`${API_URL}/projects/:id`, async ({ params }: { params: Record<string, string> }) => {
    const { id } = params
    return HttpResponse.json({
      id,
      title: 'AI in Education',
      research_question: 'How does AI affect learning?',
      keywords: ['AI', 'education', 'machine learning'],
      subtopics: ['Adaptive Learning', 'Assessment'],
      status: 'completed',
      total_papers_found: 25,
      created_at: '2024-01-15T10:00:00Z',
      synthesis_result: '# Literature Review\n\nThis is the synthesis...',
      agent_plans: [],
      paper_references: [
        {
          id: 'paper-1',
          title: 'Machine Learning in Education',
          authors: ['John Doe'],
          abstract: 'This paper explores...',
          url: 'https://arxiv.org/abs/1234.5678',
          relevance_score: 0.95,
        },
      ],
    })
  }),

  http.post(`${API_URL}/projects`, async ({ request }: { request: Request }) => {
    const body = await request.json() as { title: string; research_question: string }
    return HttpResponse.json({
      id: 'new-project-123',
      title: body.title,
      research_question: body.research_question,
      keywords: ['generated', 'keywords'],
      subtopics: ['Generated Subtopic'],
      status: 'created',
      total_papers_found: 0,
      created_at: new Date().toISOString(),
      agent_plans: [],
      paper_references: [],
    })
  }),

  http.post(`${API_URL}/projects/:id/start`, async ({ params }: { params: Record<string, string> }) => {
    const { id } = params
    return HttpResponse.json({
      job_id: 'job-123',
      status: 'queued',
      estimated_duration: 'PT5M',
    })
  }),

  // Usage endpoints
  http.get(`${API_URL}/usage/users/me/usage`, async () => {
    return HttpResponse.json({
      user_id: 'test-user-123',
      tier: 'free',
      month: '2024-01',
      budget: { limit: 1.0, used: 0.25, remaining: 0.75 },
      tokens: { total: 25000, prompt: 15000, completion: 10000 },
      activity: { projects: 2, papers: 25 },
      limits: { projects_per_month: 10, papers_per_project: 50 },
    })
  }),

  // Search endpoint
  http.post(`${API_URL}/projects/:id/search`, async () => {
    return HttpResponse.json([
      {
        chunk_id: 'chunk-1',
        content: 'Relevant content from the paper...',
        paper_id: 'paper-1',
        paper_title: 'Machine Learning in Education',
        chunk_type: 'abstract',
        score: 0.92,
      },
    ])
  }),
]

// Error handlers for testing error states
export const errorHandlers = [
  http.get(`${API_URL}/projects`, async () => {
    return HttpResponse.json(
      { detail: 'Internal server error' },
      { status: 500 }
    )
  }),

  http.post(`${API_URL}/auth/token`, async () => {
    return HttpResponse.json(
      { detail: 'Incorrect email or password' },
      { status: 401 }
    )
  }),
]

import { Routes, Route, Navigate } from 'react-router-dom'
import MainLayout from './layouts/MainLayout'
import KnowledgeBase from './pages/KnowledgeBase'
import Chat from './pages/Chat'
import Search from './pages/Search'
import Memory from './pages/Memory'
import OCR from './pages/OCR'
import './styles/pages.css'

function App() {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<Navigate to="/knowledge" replace />} />
        <Route path="knowledge" element={<KnowledgeBase />} />
        <Route path="chat" element={<Chat />} />
        <Route path="search" element={<Search />} />
        <Route path="memory" element={<Memory />} />
        <Route path="ocr" element={<OCR />} />
      </Route>
    </Routes>
  )
}

export default App

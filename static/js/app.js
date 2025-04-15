// static/js/app.js
document.addEventListener('DOMContentLoaded', function() {
    // 加载文档列表
    loadDocuments();
    
    // 设置上传表单提交事件
    document.getElementById('uploadForm').addEventListener('submit', function(e) {
        e.preventDefault();
        uploadDocument();
    });
    
    // 设置查询表单提交事件
    document.getElementById('queryForm').addEventListener('submit', function(e) {
        e.preventDefault();
        submitQuery();
    });
});

async function loadDocuments() {
    try {
        const response = await fetch('/api/list_documents');
        const data = await response.json();
        
        const tableBody = document.querySelector('#documentTable tbody');
        tableBody.innerHTML = '';
        
        if (data.documents && data.documents.length > 0) {
            data.documents.forEach(doc => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${doc.doc_id.substring(0, 8)}...</td>
                    <td>${doc.metadata.title || doc.content.substring(0, 50)}...</td>
                `;
                tableBody.appendChild(row);
            });
        } else {
            tableBody.innerHTML = '<tr><td colspan="2" class="text-center">暂无文档</td></tr>';
        }
    } catch (error) {
        console.error('加载文档列表失败:', error);
        alert('加载文档列表失败');
    }
}

async function uploadDocument() {
    const textarea = document.getElementById('documentText');
    const content = textarea.value.trim();
    
    if (!content) {
        alert('请输入文档内容');
        return;
    }
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: content,
                metadata: {
                    title: content.split('\n')[0].substring(0, 50),
                    timestamp: new Date().toISOString()
                }
            })
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            alert('文档上传成功');
            textarea.value = '';
            loadDocuments();
        } else {
            throw new Error('上传失败');
        }
    } catch (error) {
        console.error('上传文档失败:', error);
        alert('上传文档失败');
    }
}

async function submitQuery() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    
    if (!question) {
        alert('请输入问题');
        return;
    }
    
    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                top_k: 3
            })
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            displayAnswer(data.answer, data.relevant_docs);
        } else {
            throw new Error('查询失败');
        }
    } catch (error) {
        console.error('查询失败:', error);
        alert('查询失败');
    }
}

function displayAnswer(answer, relevantDocs) {
    const answerSection = document.getElementById('answerSection');
    const answerText = document.getElementById('answerText');
    const relevantDocsContainer = document.getElementById('relevantDocs');
    
    // 显示回答
    answerText.textContent = answer;
    
    // 显示相关文档
    relevantDocsContainer.innerHTML = '';
    if (relevantDocs && relevantDocs.length > 0) {
        relevantDocs.forEach(doc => {
            const docCard = document.createElement('div');
            docCard.className = 'doc-card';
            docCard.innerHTML = `
                <h6>${doc.metadata.title || '无标题'}</h6>
                <p>${doc.content.substring(0, 150)}...</p>
                <small class="text-muted">相似度: ${(doc.score * 100).toFixed(1)}%</small>
            `;
            relevantDocsContainer.appendChild(docCard);
        });
    } else {
        relevantDocsContainer.innerHTML = '<p>没有找到相关文档</p>';
    }
    
    // 显示回答区域
    answerSection.style.display = 'block';
}
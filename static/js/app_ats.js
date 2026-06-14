const { useState, useEffect, useRef } = React;

const App = () => {
    // Auth state
    const [authStep, setAuthStep] = useState('login');
    const [loginData, setLoginData] = useState({ username: '', password: '', role: '2' });
    const [twoFaData, setTwoFaData] = useState({ secret: '', qr_code: '', code: '' });

    // Dashboard state
    const [activeTab, setActiveTab] = useState('upload');
    const [results, setResults] = useState([]);
    const [history, setHistory] = useState([]);
    const [recommendations, setRecommendations] = useState([]);
    const [loading, setLoading] = useState(false);
    const [selectedIndices, setSelectedIndices] = useState([]);
    const [selectedModel, setSelectedModel] = useState('groq');

    // Structured job description form
    const [jobTitle, setJobTitle] = useState('');
    const [selectedSkills, setSelectedSkills] = useState([]);
    const [experienceYears, setExperienceYears] = useState('');
    const [remoteType, setRemoteType] = useState('On-site');
    const [files, setFiles] = useState([]);

    // Skills mapping for job titles
    const jobSkillsMapping = {
        'Software Engineer': ['Python', 'Java', 'JavaScript', 'C++', 'Go', 'Rust', 'SQL', 'Git', 'Agile', 'REST API', 'Microservices', 'Docker', 'Kubernetes', 'AWS'],
        'Frontend Developer': ['JavaScript', 'React', 'Vue', 'Angular', 'HTML', 'CSS', 'Tailwind', 'Bootstrap', 'TypeScript', 'Webpack'],
        'Backend Developer': ['Python', 'Java', 'Node.js', 'Django', 'Spring Boot', 'FastAPI', 'SQL', 'PostgreSQL', 'MongoDB', 'REST API', 'Microservices'],
        'Data Scientist': ['Python', 'Machine Learning', 'TensorFlow', 'PyTorch', 'Pandas', 'NumPy', 'SQL', 'Data Analysis', 'Statistics', 'NLP'],
        'Data Engineer': ['Python', 'SQL', 'Spark', 'Hadoop', 'Airflow', 'AWS', 'Azure', 'GCP', 'ETL', 'Data Warehousing'],
        'DevOps Engineer': ['Docker', 'Kubernetes', 'AWS', 'Terraform', 'Ansible', 'Jenkins', 'CI/CD', 'Linux', 'Python', 'Shell'],
        'Product Manager': ['Product Strategy', 'Market Research', 'Agile', 'Scrum', 'User Stories', 'Roadmap', 'Data Analysis', 'Stakeholder Management'],
        'UX Designer': ['Figma', 'Wireframing', 'Prototyping', 'User Research', 'UI Design', 'Interaction Design', 'Design Thinking', 'Adobe XD'],
        'QA Engineer': ['Test Automation', 'Selenium', 'Cypress', 'JUnit', 'TestNG', 'Manual Testing', 'Bug Tracking', 'CI/CD'],
        'Sales Manager': ['Sales Strategy', 'CRM', 'Negotiation', 'Pipeline Management', 'Forecasting', 'Customer Relations', 'Business Development'],
        'Marketing Manager': ['Digital Marketing', 'SEO', 'Content Marketing', 'Social Media', 'Google Analytics', 'Email Marketing', 'Brand Management'],
        'Financial Analyst': ['Financial Modeling', 'Excel', 'Data Analysis', 'Accounting', 'Budgeting', 'Forecasting', 'Risk Management'],
        'HR Manager': ['Recruitment', 'Onboarding', 'Performance Management', 'Employee Relations', 'HRIS', 'Payroll', 'Compliance'],
        'Project Manager': ['Agile', 'Scrum', 'Jira', 'Project Planning', 'Risk Management', 'Stakeholder Communication', 'Budget Management']
    };
    const jobTitles = Object.keys(jobSkillsMapping);

    // Update selected skills when job title changes
    useEffect(() => {
        if (jobTitle && jobSkillsMapping[jobTitle]) {
            setSelectedSkills([]);
        }
    }, [jobTitle]);

    // Build job description string from structured data
    const buildJobDescription = () => {
        let jd = `Job Title: ${jobTitle}\n`;
        jd += `Required Experience: ${experienceYears} years\n`;
        jd += `Work Type: ${remoteType}\n`;
        jd += `Required Skills: ${selectedSkills.join(', ')}\n`;
        jd += `Responsibilities: As per standard ${jobTitle} role.`;
        return jd;
    };

    // Filters for history
    const [nameFilter, setNameFilter] = useState('');
    const [scoreFilter, setScoreFilter] = useState('');
    const [jobFilter, setJobFilter] = useState('');

    const scoreChartInstance = useRef(null);
    const roleChartInstance = useRef(null);

    const filteredHistory = history.filter(c => {
        if (nameFilter && !c.name.toLowerCase().includes(nameFilter.toLowerCase())) return false;
        if (scoreFilter && c.score < parseInt(scoreFilter)) return false;
        if (jobFilter && !c.target_job.toLowerCase().includes(jobFilter.toLowerCase())) return false;
        return true;
    });

    // --- Auth Handlers ---
    const handleLogin = async (e) => {
        e.preventDefault();
        try {
            const payload = { ...loginData, role: parseInt(loginData.role, 10) };
            const res = await fetch("/login_check", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                const data = await res.json();
                if (data.status === 'enroll_2fa') {
                    setTwoFaData({ secret: data.secret, qr_code: data.qr_code, code: '' });
                    setAuthStep('2fa_enroll');
                } else if (data.status === 'verify_2fa') {
                    setTwoFaData({ secret: data.secret, qr_code: '', code: '' });
                    setAuthStep('2fa_verify');
                }
            } else {
                alert("Invalid Credentials");
            }
        } catch (e) { alert("Login failed."); }
    };

    const handle2FASubmit = async (e, isEnrollment) => {
        e.preventDefault();
        try {
            const payload = {
                username: loginData.username,
                secret: twoFaData.secret,
                code: twoFaData.code,
                is_enrollment: isEnrollment
            };
            const res = await fetch("/verify_2fa_step", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                const role = parseInt(loginData.role, 10);
                if (role === 1) {
                    window.location.href = "/admin_dashboard";
                } else if (role === 3) {
                    window.location.href = "/manager_dashboard";
                } else {
                    setAuthStep('dashboard');
                }
            } else {
                alert("Invalid 2FA Code");
            }
        } catch (e) { alert("Verification failed."); }
    };

    const handleLogout = () => {
        setAuthStep('login');
        setLoginData({ username: '', password: '', role: '2' });
        setTwoFaData({ secret: '', qr_code: '', code: '' });
        setResults([]);
        setHistory([]);
        setRecommendations([]);
    };

    // --- Data fetching ---
    const fetchHistory = async () => {
        try {
            const res = await fetch('/api/candidates');
            const data = await res.json();
            setHistory(data || []);
        } catch (e) { console.error("History error:", e); }
    };

    const fetchRecommendations = async () => {
        try {
            const res = await fetch('/api/recommendations');
            const data = await res.json();
            setRecommendations(data || []);
        } catch (e) { console.error("Recommendations error:", e); }
    };

    useEffect(() => {
        if (authStep === 'dashboard') {
            if (activeTab === 'history') fetchHistory();
            if (activeTab === 'recommendations') fetchRecommendations();
            if (activeTab === 'analytics') fetchHistory();
        }
    }, [activeTab, authStep]);

    useEffect(() => {
        if (activeTab === 'analytics' && history.length > 0 && authStep === 'dashboard') {
            setTimeout(() => renderCharts(), 100);
        }
    }, [activeTab, history, authStep]);

    const renderCharts = () => {
        if (typeof Chart === 'undefined') return;
        const scoreCanvas = document.getElementById('scoreChart');
        const roleCanvas = document.getElementById('roleChart');
        if (!scoreCanvas || !roleCanvas) return;
        if (scoreChartInstance.current) scoreChartInstance.current.destroy();
        if (roleChartInstance.current) roleChartInstance.current.destroy();

        const scoreGroups = { "90-100": 0, "70-89": 0, "50-69": 0, "<50": 0 };
        history.forEach(c => {
            if (c.score >= 90) scoreGroups["90-100"]++;
            else if (c.score >= 70) scoreGroups["70-89"]++;
            else if (c.score >= 50) scoreGroups["50-69"]++;
            else scoreGroups["<50"]++;
        });
        scoreChartInstance.current = new Chart(scoreCanvas.getContext('2d'), {
            type: 'bar',
            data: { labels: Object.keys(scoreGroups), datasets: [{ label: 'Candidates', data: Object.values(scoreGroups), backgroundColor: '#6366f1', borderRadius: 8 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });

        const roles = {};
        history.forEach(c => { roles[c.job || "Not Specified"] = (roles[c.job || "Not Specified"] || 0) + 1; });
        roleChartInstance.current = new Chart(roleCanvas.getContext('2d'), {
            type: 'doughnut',
            data: { labels: Object.keys(roles), datasets: [{ data: Object.values(roles), backgroundColor: ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'], borderWidth: 0 }] },
            options: { responsive: true, maintainAspectRatio: false, cutout: '70%' }
        });
    };

    // --- Analysis & saving ---
    const handleUpload = async () => {
        if (!jobTitle || !experienceYears || selectedSkills.length === 0) {
            alert("Please fill in job title, experience years, and at least one skill.");
            return;
        }
        if (files.length === 0) {
            alert("Please upload at least one PDF resume.");
            return;
        }
        const jd = buildJobDescription();
        setLoading(true);
        const formData = new FormData();
        formData.append("jd", jd);
        formData.append("model", selectedModel);
        for (let f of files) formData.append("files", f);
        try {
            const res = await fetch("/analyze", { method: "POST", body: formData });
            const data = await res.json();
            setResults(data);
            setActiveTab('candidates');
            setSelectedIndices([]);
        } catch (e) { alert("Analysis failed."); }
        finally { setLoading(false); }
    };

    const saveAllToHistory = async (dataList) => {
        if (dataList.length === 0) return alert("No candidates to save.");
        try {
            const res = await fetch("/api/save_candidates", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(dataList)
            });
            if (res.ok) {
                const info = await res.json();
                alert(`Saved ${info.count} candidates to history!`);
                fetchHistory();
            } else {
                alert("Save failed.");
            }
        } catch (e) { alert("Save error."); }
    };

    const saveToRecommendations = async (dataList) => {
        if (dataList.length === 0) return alert("No candidates selected.");
        try {
            const res = await fetch("/api/save_recommendations", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(dataList)
            });
            if (res.ok) {
                const info = await res.json();
                alert(`Saved ${info.count} candidates to recommendations!`);
                setSelectedIndices([]);
                if (activeTab === 'recommendations') fetchRecommendations();
            } else {
                alert("Save failed.");
            }
        } catch (e) { alert("Save error."); }
    };

    // --- Login View ---
    if (authStep === 'login') {
        return (
            <div className="h-screen flex items-center justify-center bg-gray-50 font-sans">
                <div className="bg-white p-12 rounded-[2.5rem] shadow-xl w-full max-w-md border border-gray-100">
                    <h2 className="text-3xl font-black italic text-indigo-600 mb-8 text-center tracking-tighter">ATS+</h2>
                    <form onSubmit={handleLogin} className="space-y-4">
                        <input type="text" placeholder="Username" required className="w-full p-4 bg-gray-50 rounded-2xl border-none outline-indigo-500" onChange={e => setLoginData({...loginData, username: e.target.value})} />
                        <input type="password" placeholder="Password" required className="w-full p-4 bg-gray-50 rounded-2xl border-none outline-indigo-500" onChange={e => setLoginData({...loginData, password: e.target.value})} />
                        <select className="w-full p-4 bg-gray-50 rounded-2xl border-none outline-indigo-500 text-gray-700" value={loginData.role} onChange={e => setLoginData({...loginData, role: e.target.value})}>
                            <option value="2">HR Professional</option>
                            <option value="1">Admin</option>
                            <option value="3">Head Manager</option>
                        </select>
                        <button type="submit" className="w-full bg-indigo-600 text-white p-4 rounded-2xl font-bold hover:bg-indigo-700 transition">Login</button>
                    </form>
                    <div className="flex justify-center space-x-6 pt-6 border-t border-gray-100 mt-6">
                        <a href="https://www.facebook.com/FocusCorporationIT/" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-blue-600 transition-colors duration-200" aria-label="Facebook"><i className="fab fa-facebook-f text-xl"></i></a>
                        <a href="https://focus-corporation.com" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-indigo-600 transition-colors duration-200" aria-label="Website"><i className="fas fa-globe text-xl"></i></a>
                        <a href="https://www.linkedin.com/company/focus-focus-international/" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-blue-700 transition-colors duration-200" aria-label="LinkedIn"><i className="fab fa-linkedin-in text-xl"></i></a>
                    </div>
                </div>
            </div>
        );
    }

    // 2FA views (unchanged)
    if (authStep === '2fa_enroll') {
        return (
            <div className="h-screen flex items-center justify-center bg-gray-50 font-sans">
                <div className="bg-white p-12 rounded-[2.5rem] shadow-xl w-full max-w-md border border-gray-100 flex flex-col items-center">
                    <h2 className="text-2xl font-bold text-gray-800 mb-2">Setup 2-Factor Auth</h2>
                    <p className="text-sm text-gray-500 text-center mb-6">Scan this QR code with Google Authenticator or Authy.</p>
                    <img src={twoFaData.qr_code} alt="2FA QR Code" className="w-48 h-48 mb-6 border border-gray-100 rounded-xl p-2 bg-white" />
                    <form onSubmit={(e) => handle2FASubmit(e, true)} className="w-full space-y-4">
                        <input type="text" placeholder="Enter 6-digit code" required maxLength="6" className="w-full p-4 bg-gray-50 rounded-2xl border-none outline-indigo-500 text-center tracking-widest font-mono text-lg" value={twoFaData.code} onChange={e => setTwoFaData({...twoFaData, code: e.target.value})} />
                        <button type="submit" className="w-full bg-indigo-600 text-white p-4 rounded-2xl font-bold hover:bg-indigo-700 transition shadow-lg shadow-indigo-500/30">Verify & Complete Setup</button>
                    </form>
                </div>
            </div>
        );
    }

    if (authStep === '2fa_verify') {
        return (
            <div className="h-screen flex items-center justify-center bg-gray-50 font-sans">
                <div className="bg-white p-12 rounded-[2.5rem] shadow-xl w-full max-w-md border border-gray-100 flex flex-col items-center">
                    <div className="w-16 h-16 bg-indigo-50 text-indigo-600 rounded-full flex items-center justify-center mb-6">
                        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
                    </div>
                    <h2 className="text-2xl font-bold text-gray-800 mb-2">Security Check</h2>
                    <p className="text-sm text-gray-500 text-center mb-8">Enter the 6-digit code from your authenticator app.</p>
                    <form onSubmit={(e) => handle2FASubmit(e, false)} className="w-full space-y-4">
                        <input type="text" placeholder="000000" required maxLength="6" className="w-full p-4 bg-gray-50 rounded-2xl border-none outline-indigo-500 text-center tracking-widest font-mono text-xl" value={twoFaData.code} onChange={e => setTwoFaData({...twoFaData, code: e.target.value})} />
                        <button type="submit" className="w-full bg-indigo-600 text-white p-4 rounded-2xl font-bold hover:bg-indigo-700 transition shadow-lg shadow-indigo-500/30">Authenticate</button>
                    </form>
                </div>
            </div>
        );
    }

    // --- Main Dashboard ---
    return (
        <div className="flex h-screen bg-[#fcfdfe] text-[#1a1c21] font-sans">
            {/* Sidebar */}
            <aside className="w-64 border-r border-gray-100 flex flex-col bg-white">
                <div className="p-8 text-2xl font-black italic text-indigo-600 tracking-tighter">ATS+</div>
                <nav className="flex-1 px-4 space-y-2">
                    <button onClick={() => setActiveTab('analytics')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition ${activeTab === 'analytics' ? 'bg-indigo-50 text-indigo-600' : 'text-gray-400 hover:text-indigo-600 hover:bg-gray-50'}`}><i className="fas fa-chart-line w-5"></i> Analytics</button>
                    <button onClick={() => setActiveTab('history')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition ${activeTab === 'history' ? 'bg-indigo-50 text-indigo-600' : 'text-gray-400 hover:text-indigo-600 hover:bg-gray-50'}`}><i className="fas fa-history w-5"></i> History</button>
                    <button onClick={() => setActiveTab('recommendations')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition ${activeTab === 'recommendations' ? 'bg-indigo-50 text-indigo-600' : 'text-gray-400 hover:text-indigo-600 hover:bg-gray-50'}`}><i className="fas fa-star w-5"></i> Recommendations</button>
                    <button onClick={() => setActiveTab('candidates')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition ${activeTab === 'candidates' ? 'bg-indigo-50 text-indigo-600' : 'text-gray-400 hover:text-indigo-600 hover:bg-gray-50'}`}><i className="fas fa-users w-5"></i> Candidates</button>
                    <button onClick={() => setActiveTab('upload')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition ${activeTab === 'upload' ? 'bg-indigo-50 text-indigo-600' : 'text-gray-400 hover:text-indigo-600 hover:bg-gray-50'}`}><i className="fas fa-upload w-5"></i> Upload</button>
                    <button onClick={handleLogout} className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold text-red-400 mt-10 hover:bg-red-50 hover:text-red-600 transition"><i className="fas fa-sign-out-alt w-5"></i> Logout</button>
                </nav>
            </aside>

            {/* Main Content */}
            <main className="flex-1 overflow-y-auto p-12">
                <header className="flex justify-between items-center mb-10">
                    <h1 className="text-xl font-bold uppercase tracking-widest text-gray-400">{activeTab}</h1>
                    {activeTab === 'candidates' && results.length > 0 && (
                        <div className="flex gap-4">
                            <button onClick={() => saveAllToHistory(results)} className="bg-gray-100 text-gray-600 px-6 py-2 rounded-lg font-bold text-xs uppercase hover:bg-gray-200">Save All</button>
                            <button onClick={() => saveToRecommendations(results.filter((_, i) => selectedIndices.includes(i)))} className="bg-emerald-500 text-white px-6 py-2 rounded-lg font-bold text-xs uppercase shadow-lg shadow-emerald-500/20">Recommend ({selectedIndices.length})</button>
                        </div>
                    )}
                </header>

                {/* Analytics Tab */}
                {activeTab === 'analytics' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-6xl">
                        <div className="bg-white p-10 border border-gray-100 rounded-[2.5rem] shadow-sm flex flex-col">
                            <h3 className="font-bold mb-8 text-gray-400 text-[10px] uppercase">Match Score Distribution</h3>
                            <div className="flex-1 min-h-[250px]"><canvas id="scoreChart"></canvas></div>
                        </div>
                        <div className="bg-white p-10 border border-gray-100 rounded-[2.5rem] shadow-sm flex flex-col">
                            <h3 className="font-bold mb-8 text-gray-400 text-[10px] uppercase">Candidate Roles</h3>
                            <div className="flex-1 min-h-[250px]"><canvas id="roleChart"></canvas></div>
                        </div>
                    </div>
                )}

                {/* Upload Tab (Structured Form) */}
                {activeTab === 'upload' && (
                    <div className="max-w-4xl bg-white border border-gray-100 p-8 rounded-[2rem] shadow-sm">
                        <h2 className="text-xl font-bold mb-6">Job Description Builder</h2>
                        <div className="space-y-5">
                            <div>
                                <label className="block text-sm font-bold text-gray-700 mb-1">Job Title *</label>
                                <select value={jobTitle} onChange={e => setJobTitle(e.target.value)} className="w-full p-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500">
                                    <option value="">-- Select a job title --</option>
                                    {jobTitles.map(title => <option key={title} value={title}>{title}</option>)}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-bold text-gray-700 mb-1">Required Skills *</label>
                                <select multiple value={selectedSkills} onChange={e => setSelectedSkills(Array.from(e.target.selectedOptions, o => o.value))} className="w-full p-3 border border-gray-300 rounded-xl min-h-[120px] focus:outline-none focus:ring-2 focus:ring-indigo-500" disabled={!jobTitle}>
                                    {jobTitle && jobSkillsMapping[jobTitle].map(skill => <option key={skill} value={skill}>{skill}</option>)}
                                </select>
                                <p className="text-xs text-gray-500 mt-1">Hold Ctrl (Cmd) to select multiple skills</p>
                            </div>
                            <div>
                                <label className="block text-sm font-bold text-gray-700 mb-1">Required Experience (years) *</label>
                                <input type="number" min="0" step="1" value={experienceYears} onChange={e => setExperienceYears(e.target.value)} className="w-full p-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="e.g., 5" />
                            </div>
                            <div>
                                <label className="block text-sm font-bold text-gray-700 mb-1">Work Type (optional)</label>
                                <select value={remoteType} onChange={e => setRemoteType(e.target.value)} className="w-full p-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500">
                                    <option value="On-site">On‑site</option>
                                    <option value="Hybrid">Hybrid</option>
                                    <option value="Full Remote">Full Remote</option>
                                </select>
                            </div>
                            <div className="pt-4 border-t">
                                <label className="block text-sm font-bold text-gray-700 mb-1">Resume PDFs *</label>
                                <input type="file" multiple accept=".pdf" onChange={e => setFiles(e.target.files)} className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100" />
                            </div>
                            <div className="pt-4">
                                <label className="block text-sm font-bold text-gray-700 mb-2">AI Model</label>
                                <div className="flex gap-6">
                                    <label className="flex items-center gap-2"><input type="radio" name="model" value="groq" checked={selectedModel === 'groq'} onChange={() => setSelectedModel('groq')} /> <b>Groq LLM</b> (Speed + Efficiency)</label>
                                    <label className="flex items-center gap-2"><input type="radio" name="model" value="gemini" checked={selectedModel === 'gemini'} onChange={() => setSelectedModel('gemini')} /> <b>Gemini Flash</b> (Intelligence)</label>
                                    <label className="flex items-center gap-2"><input type="radio" name="model" value="local" checked={selectedModel === 'local'} onChange={() => setSelectedModel('local')} /> <b>Local AI</b> (Secure + Private)</label>
                                </div>
                            </div>
                            <button onClick={handleUpload} disabled={loading} className="w-full mt-4 bg-indigo-600 text-white py-3 rounded-xl font-bold hover:bg-indigo-700 transition disabled:opacity-50">{loading ? 'Analyzing...' : 'Start Analysis'}</button>
                        </div>
                    </div>
                )}

                {/* Candidates Tab (Analysis Results) */}
                {activeTab === 'candidates' && (
                    <div className="space-y-8 max-w-6xl">
                        {results.length === 0 && <div className="text-gray-300 text-center py-20">No candidates. Go to Upload tab.</div>}
                        {results.map((res, i) => {
                            const getScoreLabel = (score) => {
                                if (score <= 50) return { text: 'Irrelevant', color: 'text-red-600' };
                                if (score <= 74) return { text: 'Relevant', color: 'text-yellow-500' };
                                if (score <= 85) return { text: 'Relevant', color: 'text-green-600' };
                                if (score <= 95) return { text: 'Recommended', color: 'text-green-600' };
                                return { text: 'Top candidate', color: 'text-green-600' };
                            };
                            const label = getScoreLabel(res.score);
                            return (
                                <div key={i} className="bg-white border border-gray-100 rounded-[2.5rem] p-10 shadow-sm relative">
                                    <input type="checkbox" checked={selectedIndices.includes(i)} onChange={() => setSelectedIndices(prev => prev.includes(i) ? prev.filter(x => x !== i) : [...prev, i])} className="absolute top-10 left-4 w-4 h-4 rounded" />
                                    <div className="flex justify-between items-start mb-6 pl-4">
                                        <div>
                                            <h2 className="text-2xl font-bold">{res.name}</h2>
                                            <p className="text-indigo-600 text-xs uppercase mt-1">{res.job}</p>
                                            {/* Job Recommendations Display */}
                                            {res.job_recommendations && res.job_recommendations.length > 0 && (
                                                <div className="mt-2 text-xs text-gray-600">
                                                    <div className="font-bold text-indigo-700">🎯 Top Job Match</div>
                                                    <div className="flex items-center gap-2">
                                                        <span className="bg-indigo-100 text-indigo-800 px-2 py-0.5 rounded-full">
                                                            {res.job_recommendations[0].job_title} ({res.job_recommendations[0].match_percentage}%)
                                                        </span>
                                                    </div>
                                                    {res.job_recommendations.length > 1 && (
                                                        <div className="text-gray-500 text-[11px] mt-1">
                                                            Also consider: {res.job_recommendations.slice(1,3).map(r => `${r.job_title} (${r.match_percentage}%)`).join(', ')}
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                            <div className="flex gap-4 mt-2 text-xs text-gray-500">
                                                <span className="bg-gray-100 px-2 py-1 rounded">💰 Est. Salary: ${res.suggested_salary?.toLocaleString() || 'N/A'}</span>
                                                <span className="bg-gray-100 px-2 py-1 rounded">📍 Residence: {res.residence || 'N/A'}</span>
                                            </div>
                                            <div className="flex gap-4 mt-2 text-xs text-gray-500">
                                                {res.email && res.email !== "Not found" && <span>📧 {res.email}</span>}
                                                {res.phone && res.phone !== "Not found" && <span>📞 {res.phone}</span>}
                                            </div>
                                        </div>
                                        <div className="flex flex-col items-end">
                                            <div className="text-4xl font-black text-indigo-600">{res.score}%</div>
                                            <div className={`text-xs font-bold uppercase mt-1 ${label.color}`}>{label.text}</div>
                                        </div>
                                    </div>
                                    <div className="bg-gray-50 rounded-2xl p-6 mb-6 text-sm">{res.summary}</div>
                                    <div className="grid grid-cols-2 gap-6 mb-6">
                                        <div className="bg-blue-50 p-6 rounded-2xl"><h4 className="text-[10px] font-black text-blue-600 uppercase">🎯 Top Skills</h4><div className="flex flex-wrap gap-2 mt-2">{(res.top_skills || []).map(s => <span className="bg-blue-100 px-3 py-1 rounded-full text-xs">{s}</span>)}</div></div>
                                        <div className="bg-purple-50 p-6 rounded-2xl"><h4 className="text-[10px] font-black text-purple-600 uppercase">📋 All Skills ({res.all_skills_found?.length || 0})</h4><div className="flex flex-wrap gap-2 mt-2 max-h-32 overflow-y-auto">{(res.all_skills_found || []).map(s => <span className="bg-purple-100 px-2 py-1 rounded text-xs">{s}</span>)}</div></div>
                                    </div>
                                    <div className="grid grid-cols-2 gap-6">
                                        <div className="bg-emerald-50 p-6 rounded-2xl"><h4 className="text-[10px] font-black text-emerald-600 uppercase">✅ Strengths</h4><ul className="text-xs mt-2 space-y-1">{(res.key_strengths || []).map(s => <li>• {s}</li>)}</ul></div>
                                        <div className="bg-orange-50 p-6 rounded-2xl"><h4 className="text-[10px] font-black text-orange-500 uppercase">⚠️ Gaps</h4><ul className="text-xs mt-2 space-y-1">{(res.identified_gaps || []).map(g => <li>• {g}</li>)}</ul></div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}

                {/* History Tab */}
                {activeTab === 'history' && (
                    <div>
                        <div className="bg-white p-4 rounded-xl shadow-sm mb-6 flex flex-wrap gap-4 items-end">
                            <div><label className="block text-xs font-bold text-gray-400 uppercase">Name</label><input type="text" placeholder="Search..." className="mt-1 px-3 py-2 border rounded-lg w-48" onChange={e => setNameFilter(e.target.value)} /></div>
                            <div><label className="block text-xs font-bold text-gray-400 uppercase">Min Score</label><input type="number" placeholder="Score" className="mt-1 px-3 py-2 border rounded-lg w-24" onChange={e => setScoreFilter(e.target.value)} /></div>
                            <div><label className="block text-xs font-bold text-gray-400 uppercase">JD Ref</label><input type="text" placeholder="Job ref..." className="mt-1 px-3 py-2 border rounded-lg w-48" onChange={e => setJobFilter(e.target.value)} /></div>
                            <button onClick={() => { setNameFilter(''); setScoreFilter(''); setJobFilter(''); }} className="bg-gray-200 px-4 py-2 rounded-lg text-sm">Clear</button>
                        </div>
                        <div className="bg-white border rounded-3xl overflow-hidden shadow-sm">
                            <div className="overflow-x-auto">
                                <table className="w-full text-left">
                                    <thead className="bg-gray-50"><tr className="text-[10px] uppercase text-gray-400 font-black"><th className="px-4 py-5">Name</th><th className="px-4 py-5">Exp</th><th className="px-4 py-5">Job</th><th className="px-4 py-5">Score</th><th className="px-4 py-5">JD Ref</th><th className="px-4 py-5">Suggested Job</th><th className="px-4 py-5">Salary</th><th className="px-4 py-5">Residence</th></tr></thead>
                                    <tbody>{filteredHistory.map((c,i) => (<tr key={i}><td className="px-4 py-4 font-bold">{c.name}</td><td className="px-4 py-4">{c.exp} yrs</td><td className="px-4 py-4 italic">{c.job}</td><td className="px-4 py-4 font-black text-indigo-600">{c.score}%</td><td className="px-4 py-4 text-gray-400">{c.target_job}</td><td className="px-4 py-4">{c.sug_job}</td><td className="px-4 py-4">${c.salary?.toLocaleString()}</td><td className="px-4 py-4">{c.residence}</td></tr>))}</tbody>
                                </table>
                            </div>
                            {history.length === 0 && <div className="text-center py-10 text-gray-400">No candidates in history.</div>}
                        </div>
                    </div>
                )}

                {/* Recommendations Tab */}
                {activeTab === 'recommendations' && (
                    <div className="bg-white border rounded-3xl overflow-hidden shadow-sm">
                        <div className="overflow-x-auto">
                            <table className="w-full text-left">
                                <thead className="bg-gray-50"><tr className="text-[10px] uppercase text-gray-400 font-black"><th className="px-4 py-5">Ref #</th><th className="px-4 py-5">Name</th><th className="px-4 py-5">Exp</th><th className="px-4 py-5">Job</th><th className="px-4 py-5">Score</th><th className="px-4 py-5">JD Ref</th><th className="px-4 py-5">Suggested Job</th><th className="px-4 py-5">Salary</th><th className="px-4 py-5">Residence</th><th className="px-4 py-5">Review</th><th className="px-4 py-5">Status</th></tr></thead>
                                <tbody>{recommendations.map((c,i) => (<tr key={i}><td className="px-4 py-4 font-mono">{c.rec_num}</td><td className="px-4 py-4 font-bold">{c.name}</td><td className="px-4 py-4">{c.exp} yrs</td><td className="px-4 py-4 italic">{c.job}</td><td className="px-4 py-4 font-black text-indigo-600">{c.score}%</td><td className="px-4 py-4 text-gray-400">{c.target_job}</td><td className="px-4 py-4">{c.sug_job}</td><td className="px-4 py-4">${c.salary?.toLocaleString()}</td><td className="px-4 py-4">{c.residence}</td><td className="px-4 py-4"><span className={`px-2 py-1 rounded-full text-[10px] font-bold ${c.review === 'Reviewed' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>{c.review}</span></td><td className="px-4 py-4">{c.accept === 'Accepted' ? 'Accepted' : (c.accept === 'Refused' ? 'Refused' : 'Pending')}</td></tr>))}</tbody>
                            </table>
                        </div>
                        {recommendations.length === 0 && <div className="text-center py-10 text-gray-400">No recommendations yet.</div>}
                    </div>
                )}
            </main>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
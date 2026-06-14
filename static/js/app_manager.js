const { useState, useEffect, useRef } = React;

const ManagerDashboard = () => {
    const [activeTab, setActiveTab] = useState('dashboard');
    const [metrics, setMetrics] = useState({
        totalIncome: 0,
        spendingIncome: 0,
        historyTotal: 0,
        billable: 0,
        nonBillable: 0,
        topCandidates: [],
        monthlyRevenue: [],
        categoryData: []
    });
    const [recommendations, setRecommendations] = useState([]);
    const [loading, setLoading] = useState(true);
    const [actionLoading, setActionLoading] = useState({});
    const [selectedCandidate, setSelectedCandidate] = useState(null);
    const [modalOpen, setModalOpen] = useState(false);
    const [contactModalOpen, setContactModalOpen] = useState(false);
    const [contactCandidate, setContactCandidate] = useState(null);
    const [contactEmail, setContactEmail] = useState('');
    const [contactMessage, setContactMessage] = useState('');
    const [contactSending, setContactSending] = useState(false);
    
    const revenueChartRef = useRef(null);
    const pieChartRef = useRef(null);

    const fetchDashboardData = async () => {
        try {
            const res = await fetch('/api/manager/dashboard');
            const data = await res.json();
            setMetrics(data);
        } catch (err) { console.error(err); }
    };

    const fetchRecommendations = async () => {
        try {
            const res = await fetch('/api/recommendations');
            const data = await res.json();
            setRecommendations(data);
        } catch (err) { console.error(err); }
    };

    const fetchCandidateDetails = async (rec_num) => {
        try {
            const res = await fetch(`/api/recommendations/${rec_num}/details`);
            if (res.ok) {
                const data = await res.json();
                setSelectedCandidate(data);
                setModalOpen(true);
            } else alert('Could not load details.');
        } catch (err) { alert('Error fetching details.'); }
    };

    const openContactModal = (candidate) => {
        setContactCandidate(candidate);
        setContactEmail(candidate.email || 'Not provided');
        setContactMessage('');
        setContactModalOpen(true);
    };

    const handleApproveMessage = () => {
        const name = contactCandidate?.name?.split(' ')[0] || 'Candidate';
        setContactMessage(`Dear ${name},\n\nCongratulations! Your application has been approved. We will contact you shortly.\n\nBest regards,\nHR Team`);
    };

    const handleRefuseMessage = () => {
        const name = contactCandidate?.name?.split(' ')[0] || 'Candidate';
        setContactMessage(`Dear ${name},\n\nThank you for your interest. Unfortunately, we have decided not to move forward.\n\nBest regards,\nHR Team`);
    };

    const handleConfirmSend = async () => {
        if (!contactMessage.trim()) return alert('Please enter a message.');
        setContactSending(true);
        try {
            const action = contactMessage.toLowerCase().includes('congratulations') ? 'approve' : 'refuse';
            const res = await fetch(`/api/recommendations/${contactCandidate.rec_num}/send-notification`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action, message: contactMessage })
            });
            if (res.ok) {
                alert('Notification sent!');
                setContactModalOpen(false);
            } else {
                const err = await res.json();
                alert(`Failed: ${err.detail}`);
            }
        } catch (err) { alert('Error sending.'); }
        finally { setContactSending(false); }
    };

    useEffect(() => {
        fetchDashboardData();
        fetchRecommendations();
    }, []);

    useEffect(() => {
        if (!loading && metrics.monthlyRevenue.length > 0 && activeTab === 'dashboard') {
            setTimeout(() => renderCharts(), 100);
        }
    }, [loading, metrics, activeTab]);

    const renderCharts = () => {
        const ctx1 = document.getElementById('revenueChart')?.getContext('2d');
        if (ctx1 && revenueChartRef.current) revenueChartRef.current.destroy();
        if (ctx1) {
            revenueChartRef.current = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    datasets: [{ label: 'Revenue ($)', data: metrics.monthlyRevenue, borderColor: '#4f46e5', backgroundColor: 'rgba(79,70,229,0.05)', borderWidth: 2, pointBackgroundColor: '#4f46e5', pointBorderColor: '#fff', pointRadius: 4, tension: 0.3, fill: true }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } } }
            });
        }
        const ctx2 = document.getElementById('pieChart')?.getContext('2d');
        if (ctx2 && pieChartRef.current) pieChartRef.current.destroy();
        if (ctx2 && metrics.categoryData.length) {
            pieChartRef.current = new Chart(ctx2, {
                type: 'doughnut',
                data: { labels: metrics.categoryData.map(c => c.name), datasets: [{ data: metrics.categoryData.map(c => c.value), backgroundColor: ['#4f46e5', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec489a'], borderWidth: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, cutout: '60%', plugins: { legend: { position: 'bottom' } } }
            });
        }
    };

    const handleReview = async (rec_num) => {
        setActionLoading(prev => ({ ...prev, [rec_num]: 'review' }));
        try {
            const res = await fetch(`/api/recommendations/${rec_num}/review`, { method: 'POST' });
            if (res.ok) {
                setRecommendations(prev => prev.map(r => r.rec_num === rec_num ? { ...r, review: 'Reviewed' } : r));
                alert(`Recommendation ${rec_num} reviewed.`);
            } else alert('Failed.');
        } catch (err) { alert('Error.'); }
        finally { setActionLoading(prev => ({ ...prev, [rec_num]: undefined })); }
    };

    const handleContact = (candidate) => openContactModal(candidate);
    const handleLogout = () => window.location.href = '/';
    const closeModal = () => { setModalOpen(false); setSelectedCandidate(null); };

    if (loading && metrics.historyTotal === 0 && recommendations.length === 0) return <div className="flex items-center justify-center h-screen">Loading dashboard...</div>;

    return (
        <div className="min-h-screen bg-gray-100 flex">
            {/* Sidebar */}
            <aside className="w-64 bg-white shadow-sm border-r flex flex-col">
                <div className="p-6 text-2xl font-bold text-indigo-600 border-b">ATS+ Manager</div>
                <nav className="flex-1 px-4 py-6 space-y-2">
                    <button onClick={() => setActiveTab('dashboard')} className={`w-full text-left px-4 py-3 rounded-xl text-sm font-semibold transition flex items-center gap-3 ${activeTab === 'dashboard' ? 'bg-indigo-50 text-indigo-600' : 'text-gray-600 hover:bg-gray-50'}`}>
                        <i className="fas fa-chart-pie w-4"></i> Dashboard
                    </button>
                    <button onClick={() => setActiveTab('review')} className={`w-full text-left px-4 py-3 rounded-xl text-sm font-semibold transition flex items-center gap-3 ${activeTab === 'review' ? 'bg-indigo-50 text-indigo-600' : 'text-gray-600 hover:bg-gray-50'}`}>
                        <i className="fas fa-check-circle w-4"></i> Review Recommendations
                    </button>
                </nav>
                <div className="p-4 border-t">
                    <button onClick={handleLogout} className="w-full text-left text-sm text-red-500 hover:text-red-700 flex items-center gap-2">
                        <i className="fas fa-sign-out-alt"></i> Logout
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 overflow-y-auto p-6">
                {activeTab === 'dashboard' && (
                    <>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                            <div className="bg-white rounded-2xl shadow-sm p-6 card"><div className="text-gray-400 text-sm uppercase">Total Income</div><div className="text-3xl font-bold">${metrics.totalIncome.toLocaleString()}</div><div className="text-green-500 text-sm">↑ 12% from last month</div></div>
                            <div className="bg-white rounded-2xl shadow-sm p-6 card"><div className="text-gray-400 text-sm uppercase">Spending Income</div><div className="text-3xl font-bold">${metrics.spendingIncome.toLocaleString()}</div><div className="text-red-500 text-sm">↓ 4% from last month</div></div>
                            <div className="bg-white rounded-2xl shadow-sm p-6 card"><div className="text-gray-400 text-sm uppercase">History Total</div><div className="text-3xl font-bold">{metrics.historyTotal}</div><div className="text-gray-400 text-sm">Candidates processed</div></div>
                            <div className="bg-white rounded-2xl shadow-sm p-6 card"><div className="text-gray-400 text-sm uppercase">Utilization Rate</div><div className="flex items-end gap-2"><span className="text-3xl font-bold">{metrics.billable}%</span><span className="text-gray-400 text-sm">billable</span></div><div className="w-full bg-gray-200 rounded-full h-2 mt-2"><div className="bg-indigo-600 h-2 rounded-full" style={{ width: `${metrics.billable}%` }}></div></div><div className="text-gray-400 text-xs mt-2">{metrics.nonBillable}% non-billable</div></div>
                        </div>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                            <div className="bg-white rounded-2xl shadow-sm p-6"><h3 className="font-bold text-gray-700 mb-4">Monthly Revenue</h3><div className="h-64"><canvas id="revenueChart"></canvas></div></div>
                            <div className="bg-white rounded-2xl shadow-sm p-6"><h3 className="font-bold text-gray-700 mb-4">Category Distribution</h3><div className="h-64"><canvas id="pieChart"></canvas></div></div>
                        </div>
                        <div className="bg-white rounded-2xl shadow-sm overflow-hidden">
                            <div className="px-6 py-4 border-b"><h3 className="font-bold text-gray-700">Top Candidates (by Match Score)</h3></div>
                            <div className="overflow-x-auto"><table className="w-full text-left"><thead className="bg-gray-50 text-xs text-gray-400 uppercase"><tr><th className="px-6 py-3">Name</th><th className="px-6 py-3">Job</th><th className="px-6 py-3">Score</th><th className="px-6 py-3">Experience</th></tr></thead><tbody>{metrics.topCandidates.map((c,idx) => (<tr key={idx}><td className="px-6 py-3 font-medium">{c.name}</td><td className="px-6 py-3 text-gray-500">{c.job}</td><td className="px-6 py-3 font-bold text-indigo-600">{c.score}%</td><td className="px-6 py-3 text-gray-500">{c.exp} yrs</td></tr>))}</tbody></table></div>
                        </div>
                    </>
                )}

                {activeTab === 'review' && (
                    <div className="bg-white rounded-2xl shadow-sm overflow-hidden">
                        <div className="px-6 py-4 border-b"><h3 className="font-bold text-gray-700">Recommendations for Review</h3><p className="text-xs text-gray-400">Click on any row to view full details.</p></div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-left">
                                <thead className="bg-gray-50 text-xs text-gray-400 uppercase">
                                    <tr>
                                        <th className="px-4 py-3">Ref #</th>
                                        <th className="px-4 py-3">Name</th>
                                        <th className="px-4 py-3">Exp</th>
                                        <th className="px-4 py-3">Job</th>
                                        <th className="px-4 py-3">Score</th>
                                        <th className="px-4 py-3">JD Ref</th>
                                        <th className="px-4 py-3">Review Status</th>
                                        <th className="px-4 py-3">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y">
                                    {recommendations.map((rec) => (
                                        <tr key={rec.rec_num} className="text-sm cursor-pointer hover:bg-gray-50" onClick={() => fetchCandidateDetails(rec.rec_num)}>
                                            <td className="px-4 py-3 font-mono text-xs">{rec.rec_num}</td>
                                            <td className="px-4 py-3 font-medium">{rec.name}</td>
                                            <td className="px-4 py-3">{rec.exp} yrs</td>
                                            <td className="px-4 py-3 text-gray-500">{rec.job || 'N/A'}</td>
                                            <td className="px-4 py-3 font-bold text-indigo-600">{rec.score}%</td>
                                            <td className="px-4 py-3 text-gray-400 truncate max-w-xs">{rec.target_job}</td>
                                            <td className="px-4 py-3">
                                                <span className={`px-2 py-1 rounded-full text-[10px] font-bold ${rec.review === 'Reviewed' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                                                    {rec.review}
                                                </span>
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
                                                <button onClick={() => handleReview(rec.rec_num)} disabled={rec.review === 'Reviewed'} className="mr-2 px-3 py-1 rounded text-xs font-medium bg-indigo-100 text-indigo-700 hover:bg-indigo-200 disabled:opacity-50">Reviewed</button>
                                                <button onClick={() => handleContact(rec)} className="px-3 py-1 rounded text-xs font-medium bg-emerald-100 text-emerald-700 hover:bg-emerald-200">Contact Candidate</button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        {recommendations.length === 0 && <div className="text-center py-10 text-gray-400">No recommendations available.</div>}
                    </div>
                )}
            </main>

            {/* Modal for candidate details (unchanged) */}
            {modalOpen && selectedCandidate && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full">
                        <div className="sticky top-0 bg-white border-b px-6 py-4 flex justify-between"><div><h2 className="text-xl font-bold">{selectedCandidate.name}</h2><p className="text-sm text-indigo-600">{selectedCandidate.job} • Score: {selectedCandidate.score}%</p></div><button onClick={closeModal} className="text-gray-400 hover:text-gray-600 text-2xl">&times;</button></div>
                        <div className="p-6 space-y-6">
                            <div><h3 className="font-bold text-emerald-600">✅ Strengths</h3><ul className="list-disc pl-5">{(selectedCandidate.strengths||[]).map((s,i)=><li key={i}>{s}</li>)}</ul></div>
                            <div><h3 className="font-bold text-orange-500">⚠️ Gaps</h3><ul className="list-disc pl-5">{(selectedCandidate.gaps||[]).map((g,i)=><li key={i}>{g}</li>)}</ul></div>
                            <div><h3 className="font-bold text-blue-600">📋 Skills</h3><div className="flex flex-wrap gap-2">{(selectedCandidate.skills||[]).map(s=><span className="bg-blue-100 px-2 py-1 rounded text-xs">{s}</span>)}</div></div>
                        </div>
                        <div className="border-t px-6 py-4 flex justify-end"><button onClick={closeModal} className="px-4 py-2 bg-gray-200 rounded-lg">Close</button></div>
                    </div>
                </div>
            )}

            {/* Contact modal (unchanged) */}
            {contactModalOpen && contactCandidate && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-lg w-full">
                        <div className="border-b px-6 py-4 flex justify-between"><h2 className="text-xl font-bold">Contact Candidate</h2><button onClick={() => setContactModalOpen(false)} className="text-gray-400 hover:text-gray-600 text-2xl">&times;</button></div>
                        <div className="p-6 space-y-4">
                            <div><label className="block text-sm font-medium text-gray-700">Email</label><input type="email" value={contactEmail} disabled className="w-full border rounded-lg px-3 py-2 bg-gray-50" /></div>
                            <div><label className="block text-sm font-medium text-gray-700">Message</label><textarea rows="6" value={contactMessage} onChange={e => setContactMessage(e.target.value)} className="w-full border rounded-lg px-3 py-2"></textarea></div>
                            <div className="flex justify-end gap-3">
                                <button onClick={() => setContactModalOpen(false)} className="px-4 py-2 border rounded-lg">Cancel</button>
                                <button onClick={handleApproveMessage} className="px-4 py-2 bg-green-600 text-white rounded-lg">Approve</button>
                                <button onClick={handleRefuseMessage} className="px-4 py-2 bg-red-600 text-white rounded-lg">Refuse</button>
                                <button onClick={handleConfirmSend} disabled={contactSending} className="px-4 py-2 bg-indigo-600 text-white rounded-lg">{contactSending ? 'Sending...' : 'Confirm'}</button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('manager-root'));
root.render(<ManagerDashboard />);
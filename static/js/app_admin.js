const { useState, useEffect } = React;

const AdminApp = () => {
    const [activeTab, setActiveTab] = useState('dashboard');
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(false);
    
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalMode, setModalMode] = useState('edit'); 
    const [formData, setFormData] = useState({
        usr_name: '', usr_last_name: '', role: '2', usr_password: '', lock: false, clear_2fa: false
    });

    useEffect(() => { fetchUsers(); }, []);

    const fetchUsers = async () => {
        setLoading(true);
        try {
            const res = await fetch("/api/admin/users");
            if (res.ok) {
                const data = await res.json();
                setUsers(data);
            }
        } catch (err) { console.error("Fetch error:", err); }
        finally { setLoading(false); }
    };

    const openAddModal = () => {
        setModalMode('add');
        setFormData({ usr_name: '', usr_last_name: '', role: '2', usr_password: '', lock: false, clear_2fa: false });
        setIsModalOpen(true);
    };

    const openEditModal = (user) => {
        setModalMode('edit');
        setFormData({ ...user, usr_password: '', clear_2fa: false });
        setIsModalOpen(true);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const url = modalMode === 'add' ? '/api/admin/create_user' : `/api/admin/update_user/${formData.id}`;
        try {
            const res = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData)
            });
            if (res.ok) {
                setIsModalOpen(false);
                fetchUsers();
                alert(modalMode === 'add' ? "User created successfully!" : "User updated successfully!");
            }
        } catch (err) { alert("Operation failed."); }
    };

    const stats = {
        total: users.length,
        admins: users.filter(u => u.role === 1).length,
        active: users.filter(u => !u.lock).length,
        locked: users.filter(u => u.lock).length
    };

    return (
        <div className="flex h-screen font-sans overflow-hidden">
            {/* Sidebar */}
            <aside className="w-64 sidebar-dark text-[#b8c7ce] flex flex-col shadow-xl">
                <div className="h-14 bg-[#367fa9] flex items-center justify-center text-white font-bold text-xl uppercase tracking-tighter">
                    ATS+ <span className="font-light ml-1 italic text-sm">Admin</span>
                </div>
                <nav className="flex-1 mt-4 text-xs">
                    <button onClick={() => setActiveTab('dashboard')} className={`w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-[#1e282c] ${activeTab === 'dashboard' ? 'nav-active' : ''}`}>
                        <i className="fas fa-tachometer-alt w-4"></i> Dashboard
                    </button>
                    <button onClick={() => setActiveTab('users')} className={`w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-[#1e282c] ${activeTab === 'users' ? 'nav-active' : ''}`}>
                        <i className="fas fa-users-cog w-4"></i> User Management
                    </button>
                    <button onClick={() => window.location.href = '/'} className="w-full text-left px-4 py-3 flex items-center gap-3 text-red-400 mt-auto mb-4">
                        <i className="fas fa-power-off w-4"></i> Log Out
                    </button>
                </nav>
            </aside>

            {/* Main Content */}
            <main className="flex-1 flex flex-col min-w-0">
                <header className="bg-[#3c8dbc] h-14 flex items-center px-4 shadow text-white">
                    <h2 className="text-lg font-light">Admin Control Panel</h2>
                </header>

                <div className="flex-1 overflow-y-auto p-5">
                    {activeTab === 'dashboard' && (
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
                            <div className="small-box bg-aqua shadow-lg"><div className="inner"><h3 className="text-3xl font-bold">{stats.total}</h3><p>Total Users</p></div><div className="icon"><i className="fas fa-users"></i></div></div>
                            <div className="small-box bg-yellow shadow-lg"><div className="inner"><h3 className="text-3xl font-bold">{stats.admins}</h3><p>Admins</p></div><div className="icon"><i className="fas fa-shield-alt"></i></div></div>
                            <div className="small-box bg-green shadow-lg"><div className="inner"><h3 className="text-3xl font-bold">{stats.active}</h3><p>Active</p></div><div className="icon"><i className="fas fa-user-check"></i></div></div>
                            <div className="small-box bg-red shadow-lg"><div className="inner"><h3 className="text-3xl font-bold">{stats.locked}</h3><p>Locked</p></div><div className="icon"><i className="fas fa-user-lock"></i></div></div>
                        </div>
                    )}

                    {activeTab === 'users' && (
                        <div className="bg-white rounded shadow-sm border-t-4 border-[#3c8dbc]">
                            <div className="p-4 border-b font-bold text-gray-700 flex justify-between items-center">
                                <span><i className="fas fa-users-cog mr-2"></i> User Registry</span>
                                <div className="flex gap-2">
                                    <button onClick={openAddModal} className="text-[10px] bg-green-600 text-white px-3 py-1.5 rounded font-bold hover:bg-green-700 transition flex items-center gap-1 shadow-sm uppercase">
                                        <i className="fas fa-user-plus"></i> Add User
                                    </button>
                                    <button onClick={fetchUsers} className="text-[10px] bg-gray-100 px-3 py-1.5 rounded border hover:bg-gray-200"><i className="fas fa-sync-alt"></i></button>
                                </div>
                            </div>
                            <div className="p-4 overflow-x-auto">
                                <table className="w-full text-xs text-left">
                                    <thead>
                                        <tr className="bg-gray-50 text-gray-400 uppercase text-[10px] font-black border-b">
                                            <th className="p-4">Full Name</th><th className="p-4">Role</th><th className="p-4 text-center">2FA</th><th className="p-4 text-center">Status</th><th className="p-4 text-center">Action</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y text-[11px]">
                                        {users.map((u) => (
                                            <tr key={u.id} className="hover:bg-blue-50/20">
                                                <td className="p-4 font-bold uppercase">{u.usr_name} {u.usr_last_name}</td>
                                                <td className="p-4 italic text-blue-600 font-bold">{u.role_name}</td>
                                                <td className="p-4 text-center">{u.has_2fa ? '✅' : '❌'}</td>
                                                <td className="p-4 text-center">
                                                    <span className={`px-3 py-1 rounded text-white font-black text-[8px] uppercase ${u.lock ? 'bg-red' : 'bg-green'}`}>
                                                        {u.lock ? 'Locked' : 'Active'}
                                                    </span>
                                                </td>
                                                <td className="p-4 text-center">
                                                    <button onClick={() => openEditModal(u)} className="bg-[#3c8dbc] text-white px-3 py-1 rounded text-[9px] font-bold shadow-sm uppercase">Edit</button>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            </main>

            {/* Modal (unchanged) */}
            {isModalOpen && (
                <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 backdrop-blur-sm p-4">
                    <div className="bg-white rounded-lg shadow-2xl w-full max-w-md border-t-8 border-[#3c8dbc] overflow-hidden">
                        <div className="p-6">
                            <h2 className="text-lg font-bold text-gray-800 mb-6 uppercase tracking-wider">
                                {modalMode === 'add' ? "Create User" : "Modify User"}
                            </h2>
                            <form onSubmit={handleSubmit} className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="text-[10px] font-black text-gray-400 block uppercase mb-1">First Name</label>
                                        <input required type="text" className="w-full border rounded p-2 text-xs" value={formData.usr_name} onChange={e => setFormData({...formData, usr_name: e.target.value})} />
                                    </div>
                                    <div>
                                        <label className="text-[10px] font-black text-gray-400 block uppercase mb-1">Last Name</label>
                                        <input required type="text" className="w-full border rounded p-2 text-xs" value={formData.usr_last_name} onChange={e => setFormData({...formData, usr_last_name: e.target.value})} />
                                    </div>
                                </div>
                                <div>
                                    <label className="text-[10px] font-black text-gray-400 block uppercase mb-1">System Role</label>
                                    <select className="w-full border rounded p-2 text-xs font-bold bg-gray-50" value={formData.role} onChange={e => setFormData({...formData, role: e.target.value})}>
                                        <option value="1">Administrator</option>
                                        <option value="2">HR Professional</option>
                                        <option value="3">Head Manager</option>
                                    </select>
                                </div>
                                {modalMode === 'add' && (
                                    <div>
                                        <label className="text-[10px] font-black text-gray-400 block uppercase mb-1">Account Password</label>
                                        <input required type="password" placeholder="Enter secure password" className="w-full border rounded p-2 text-xs" value={formData.usr_password} onChange={e => setFormData({...formData, usr_password: e.target.value})} />
                                    </div>
                                )}
                                {modalMode === 'edit' && (
                                    <div className="space-y-3 pt-2">
                                        <div className="flex items-center gap-3 p-3 bg-red-50 border border-red-100 rounded text-red-700">
                                            <input type="checkbox" id="c2fa" checked={formData.clear_2fa} onChange={e => setFormData({...formData, clear_2fa: e.target.checked})} />
                                            <label htmlFor="c2fa" className="text-[10px] font-bold uppercase cursor-pointer">Reset 2FA Key</label>
                                        </div>
                                        <div className="flex items-center gap-3 p-3 bg-gray-50 border rounded">
                                            <input type="checkbox" id="lockU" checked={formData.lock} onChange={e => setFormData({...formData, lock: e.target.checked})} />
                                            <label htmlFor="lockU" className="text-[10px] font-bold uppercase cursor-pointer">Lock Account</label>
                                        </div>
                                    </div>
                                )}
                                <div className="flex justify-end gap-3 mt-6 pt-4 border-t">
                                    <button type="button" onClick={() => setIsModalOpen(false)} className="text-[11px] font-bold text-gray-400 px-4 hover:text-gray-600 uppercase">Cancel</button>
                                    <button type="submit" className={`px-6 py-2 rounded text-[11px] font-bold text-white shadow-md uppercase ${modalMode === 'add' ? 'bg-green-600' : 'bg-[#3c8dbc]'}`}>
                                        {modalMode === 'add' ? 'Create User' : 'Save Changes'}
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('admin-root'));
root.render(<AdminApp />);
const { useState } = React;

const App = () => {
    const [view, setView] = useState('Login');
    const [authData, setAuthData] = useState(null);
    const [otp, setOtp] = useState("");
    const [creds, setCreds] = useState({ username: '', password: '', role: 2 });

    const handleLogin = async () => {
        const res = await fetch("/login_check", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(creds)
        });
        if (res.ok) {
            setAuthData(await res.json());
            setView('2FA');
        } else { alert("Access Denied."); }
    };

    const handleVerify2FA = async () => {
        const res = await fetch("/verify_2fa_step", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                username: authData.username, secret: authData.secret,
                code: otp, is_enrollment: authData.status === 'enroll_2fa'
            })
        });
        if (res.ok) {
            // Updated to match the unified dashboard
            window.location.href = "/ats_main.html";
        } else { alert("Invalid Code."); }
    };

    if (view === 'Login') return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50">
            <div className="bg-white p-10 rounded-[40px] shadow-2xl shadow-indigo-100 w-full max-w-sm">
                <h1 className="text-3xl font-black text-indigo-600 mb-8 text-center tracking-tighter">UPLEE</h1>
                <input className="w-full p-4 bg-slate-50 border rounded-2xl mb-4 outline-none focus:ring-2 ring-indigo-500" placeholder="Username" onChange={e => setCreds({...creds, username: e.target.value})} />
                <input className="w-full p-4 bg-slate-50 border rounded-2xl mb-4 outline-none focus:ring-2 ring-indigo-500" type="password" placeholder="Password" onChange={e => setCreds({...creds, password: e.target.value})} />
                <button onClick={handleLogin} className="w-full py-4 bg-indigo-600 text-white rounded-2xl font-bold shadow-lg shadow-indigo-100 hover:scale-[1.02] transition-all">Sign In</button>
            </div>
        </div>
    );

    return (
        <div className="min-h-screen flex items-center justify-center bg-indigo-600">
            <div className="bg-white p-10 rounded-[40px] w-full max-w-sm text-center">
                <h2 className="text-xl font-bold mb-6">Security Code</h2>
                {authData.status === 'enroll_2fa' && <img src={authData.qr_code} className="mx-auto mb-6 w-44 rounded-2xl border-4 border-slate-50" />}
                <input className="w-full p-4 bg-slate-100 rounded-2xl mb-6 text-center text-2xl tracking-[10px] font-black" maxLength="6" onChange={e => setOtp(e.target.value)} />
                <button onClick={handleVerify2FA} className="w-full py-4 bg-slate-900 text-white rounded-2xl font-bold hover:bg-black transition-all">Unlock Dashboard</button>
            </div>
        </div>
    );
};

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
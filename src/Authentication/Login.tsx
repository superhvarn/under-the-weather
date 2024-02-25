import React, {useState} from "react"

export const Login = (props: any) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = async (e: any) => {
        e.preventDefault();
        try {
            const response = await fetch('http://127.0.0.1:5000/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password }),
            });
            if (response.ok) {
                // Login successful
                // Redirect to dashboard or perform other actions
                console.log('Login successful');
            } else {
                const data = await response.json();
                console.error("Error:");
            }
        } catch (error) {
            console.error('Error:', error);
        }
        props.onFormSwitch('state');
    }

    const handleBack = (e: any) => {
        e.preventDefault();
        props.onFormSwitch('home');
    }

    return (
        <div className={"auth-form-container"}>
            <form className={"login-form"} onSubmit={handleSubmit}>
                <label htmlFor={"email"}>Email</label>
                <input value={email} onChange={(e) => setEmail(e.target.value)}
                       type={"email"} placeholder={"your-email@gmail.com"} id={"email"} name={"email"}/>

                <label htmlFor={"password"}>Password</label>
                <input value={password} onChange={(e) => setPassword(e.target.value)}
                       type={"password"} placeholder={"your-password!@#"} id={"password"} name={"password"}/>

                <button type={"submit"}>Log In</button>
            </form>
            <button type={"reset"} className={"link-btn"} onClick={() => props.onFormSwitch('register')}>
                Don't have an account? Register here:
            </button>
            <button onClick={handleBack}>Back</button>
        </div>
    )
}
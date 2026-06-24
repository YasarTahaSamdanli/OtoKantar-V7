<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\User;
use App\Services\AuditLogService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;
use Illuminate\Validation\Rules\Password;

class UserController extends Controller
{
    public function __construct(
        private readonly AuditLogService $audit,
    ) {}

    public function index()
    {
        return view('admin.users.index', [
            'users' => User::query()->orderBy('id', 'desc')->paginate(20),
        ]);
    }

    public function create()
    {
        return view('admin.users.create');
    }

    public function store(Request $request)
    {
        $data = $request->validate([
            'name' => ['required', 'string', 'max:255'],
            'email' => ['required', 'string', 'lowercase', 'email', 'max:255', 'unique:'.User::class],
            'password' => ['required', 'confirmed', Password::defaults()],
            'role' => ['required', 'string', 'in:admin,employee'],
        ]);

        $user = User::create([
            'name' => $data['name'],
            'email' => $data['email'],
            'password' => Hash::make($data['password']),
            'role' => $data['role'],
        ]);

        $this->audit->record('admin.user.created', $request, $user, [
            'created_user_email' => $user->email,
            'created_user_role' => $user->role,
        ]);

        return redirect()
            ->route('admin.users.index')
            ->with('status', 'Kullanici olusturuldu.');
    }
}
